import base64
import json
import os
import urllib.parse
from pathlib import Path

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes  # type: ignore[reportMissingImports]
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

# Attempt to import Google libraries
try:
    import hashlib
    import time

    from google import genai
    from google.cloud import aiplatform, storage
    from google.oauth2 import service_account

    GOOGLE_INSTALLED = True
except ImportError:
    GOOGLE_INSTALLED = False


class BaseAnalyzeMedia(ControlNode):
    # Service constants for configuration
    SERVICE = "GoogleAI"
    SERVICE_ACCOUNT_FILE_PATH = "GOOGLE_SERVICE_ACCOUNT_FILE_PATH"
    PROJECT_ID = "GOOGLE_CLOUD_PROJECT_ID"
    CREDENTIALS_JSON = "GOOGLE_APPLICATION_CREDENTIALS_JSON"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "Media Analysis/Google AI"
        self.description = "Analyzes images, videos, or audio and answers questions about the media content using Google's Gemini model."

        # Main Parameters
        self.add_parameter(
            Parameter(
                name="prompt",
                type="str",
                tooltip="The prompt/question to ask about the media content.",
                ui_options={"multiline": True, "placeholder_text": "What would you like to know about this media?"},
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            ParameterList(
                name="media",
                input_types=[
                    "VideoUrlArtifact",
                    "ImageArtifact",
                    "ImageUrlArtifact",
                    "AudioArtifact",
                    "AudioUrlArtifact",
                    "Any",
                    "any",
                ],
                type="VideoUrlArtifact",
                tooltip="The media artifact to analyze (image, video, or audio).",
                allowed_modes={ParameterMode.INPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="model",
                type="str",
                tooltip="The Gemini model to use for analysis.",
                default_value="gemini-2.5-flash",
                traits=[
                    Options(choices=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite"])
                ],
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="temperature",
                type="float",
                tooltip="Controls randomness in the response (0.0 = deterministic, 1.0 = very random).",
                default_value=0.4,
                traits={Slider(min_val=0.0, max_val=1.0)},
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={"hide": True},
            )
        )

        self.add_parameter(
            Parameter(
                name="max_tokens",
                type="int",
                tooltip="Maximum number of tokens in the response.",
                default_value=2048,
                traits=[Options(choices=[512, 1024, 2048, 4096, 8192])],
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={"hide": True},
            )
        )

        self.add_parameter(
            Parameter(
                name="location",
                type="str",
                tooltip="Google Cloud location for the generation job.",
                default_value="us-central1",
                traits=[
                    Options(
                        choices=["us-central1", "us-east1", "us-west1", "europe-west1", "europe-west4", "asia-east1"]
                    )
                ],
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={"hide": True},
            )
        )

        # Output Parameters
        self.add_parameter(
            Parameter(
                name="output",
                type="str",
                tooltip="The AI's response describing or answering questions about the media.",
                ui_options={"multiline": True, "placeholder_text": "AI response will appear here"},
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="media_count",
                type="int",
                tooltip="The number of media items that were processed.",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="media_type",
                type="str",
                tooltip="The detected type of media (image, video, or audio).",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        # Logs Group
        with ParameterGroup(name="Logs") as logs_group:
            Parameter(
                name="logs",
                type="str",
                tooltip="Logs from the media analysis process.",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"multiline": True, "placeholder_text": "Logs"},
            )

        # logs_group.ui_options = {"hide": True}
        self.add_node_element(logs_group)

    def _log(self, message: str) -> None:
        """Append a message to the logs output parameter."""
        self.append_value_to_parameter("logs", message + "\n")

    def _raise_file_not_found(self, file_path: str) -> None:
        """Raise FileNotFoundError with logging."""
        msg = f"Local file not found: {file_path}"
        self._log(msg)
        raise FileNotFoundError(msg)

    def _get_project_id(self, service_account_file: str) -> str:
        """Read the project_id from the service account JSON file."""
        if not Path(service_account_file).exists():
            msg = f"Service account file not found: {service_account_file}"
            self._log(msg)
            raise FileNotFoundError(msg)

        with Path(service_account_file).open() as f:
            service_account_info = json.load(f)

        project_id = service_account_info.get("project_id")
        if not project_id:
            msg = "No 'project_id' found in the service account file."
            self._log(msg)
            raise ValueError(msg)

        return project_id

    def _upload_to_gcs(  # noqa: PLR0913
        self, media_data: bytes, filename: str, mime_type: str, project_id: str, credentials: any, location: str
    ) -> str:
        """Upload media file to GCS bucket and return the GCS URI."""
        try:
            # Use the bucket name that matches your setup
            bucket_name = "griptape-nodes"

            # Initialize storage client
            storage_client = storage.Client(project=project_id, credentials=credentials)

            # Get the bucket
            bucket = storage_client.bucket(bucket_name)

            # Check if file already exists
            blob_path = f"media/{filename}"
            blob = bucket.blob(blob_path)

            if blob.exists():
                self._log(f"📁 File already exists in GCS: {filename}")
                gcs_uri = f"gs://{bucket_name}/{blob_path}"
                self._log(f"✅ Using existing file: {gcs_uri}")
                return gcs_uri
            self._log(f"📤 Uploading {filename} to GCS bucket...")

            # Create blob and upload
            blob.upload_from_string(media_data, content_type=mime_type)

            # For uniform bucket-level access, we don't need to make individual objects public
            # The bucket-level permissions will handle access
            gcs_uri = f"gs://{bucket_name}/{blob_path}"
            self._log(f"✅ File uploaded successfully: {gcs_uri}")

            return gcs_uri

        except Exception as e:
            self._log(f"❌ Error uploading to GCS: {e}")
            self._log(
                "💡 Make sure you have a bucket named 'griptape-nodes' in the same region as your Vertex AI setup."
            )
            raise

    def _get_media_source(self, media_artifact: any) -> dict:
        """Get media source information for processing."""
        self._log("🔄 Processing media artifact...")

        # Check if it's a public URL (not localhost)
        if hasattr(media_artifact, "value") and isinstance(media_artifact.value, str):
            if "localhost" in media_artifact.value or "127.0.0.1" in media_artifact.value:
                return {"type": "localhost_url", "url": media_artifact.value}
            return {"type": "public_url", "url": media_artifact.value}

        # Direct artifact
        return {"type": "direct_artifact", "artifact": media_artifact}

    def _extract_bytes_from_artifact(self, media_artifact: any) -> bytes:
        """Extract bytes from any media artifact."""
        if hasattr(media_artifact, "value") and hasattr(media_artifact.value, "read"):
            # File-like object
            return media_artifact.value.read()
        if hasattr(media_artifact, "base64"):
            # Base64 encoded
            return base64.b64decode(media_artifact.base64)
        # Direct bytes or other format
        return media_artifact.value

    def _get_localhost_file_path(self, url: str) -> Path:
        """Convert localhost URL to local file path."""
        parsed_url = urllib.parse.urlparse(url)
        filename = parsed_url.path.split("/")[-1].split("?")[0]  # Remove query params
        static_files_path = GriptapeNodes.StaticFilesManager()._get_static_files_directory()
        print(static_files_path)
        full_path = GriptapeNodes.ConfigManager().workspace_path / static_files_path

        return full_path / filename

    def _generate_filename(self, media_artifact: any, content_hash: str) -> str:
        """Generate filename with original name + content hash."""
        if hasattr(media_artifact, "value") and isinstance(media_artifact.value, str):
            # URL artifact - extract filename
            import urllib.parse

            parsed_url = urllib.parse.urlparse(media_artifact.value)
            original_name = parsed_url.path.split("/")[-1].split("?")[0]
        else:
            # Direct artifact - use name or default
            original_name = getattr(media_artifact, "name", "media")

        # Get extension from original name
        if "." in original_name:
            name, extension = original_name.rsplit(".", 1)
        else:
            name, extension = original_name, "bin"

        return f"{name}_{content_hash[:8]}.{extension}"

    def _get_mime_type(self, filename: str) -> str:
        """Get MIME type from filename extension."""
        extension = filename.lower().split(".")[-1]

        mime_types = {
            # Images
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
            # Videos
            "mp4": "video/mp4",
            "webm": "video/webm",
            "avi": "video/avi",
            "mov": "video/quicktime",
            # Audio
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "ogg": "audio/ogg",
        }

        return mime_types.get(extension, "application/octet-stream")

    def _process_media_artifact(
        self, media_artifact: any, project_id: str = None, credentials=None, location: str = None
    ) -> dict:
        """Process a single media artifact and return source info."""
        source = self._get_media_source(media_artifact)

        if source["type"] == "public_url":
            # Public URL - use directly
            self._log(f"🌐 Using public URL: {source['url']}")
            return {"type": "url", "value": source["url"]}

        if source["type"] == "localhost_url":
            # Localhost URL - read file and upload to GCS
            local_path = self._get_localhost_file_path(source["url"])

            if not local_path.exists():
                self._raise_file_not_found(local_path)

            self._log(f"📁 Reading local file: {local_path}")
            with local_path.open("rb") as f:
                media_data = f.read()

            # Generate filename and upload to GCS
            content_hash = hashlib.md5(media_data).hexdigest()
            filename = self._generate_filename(media_artifact, content_hash)
            mime_type = self._get_mime_type(filename)

            gcs_uri = self._upload_to_gcs(media_data, filename, mime_type, project_id, credentials, location)
            return {"type": "gcs", "value": gcs_uri, "mime_type": mime_type}

        # Direct artifact
        # Extract bytes and upload to GCS
        media_data = self._extract_bytes_from_artifact(media_artifact)

        # Generate filename and upload to GCS
        content_hash = hashlib.md5(media_data).hexdigest()
        filename = self._generate_filename(media_artifact, content_hash)
        mime_type = self._get_mime_type(filename)

        gcs_uri = self._upload_to_gcs(media_data, filename, mime_type, project_id, credentials, location)
        return {"type": "gcs", "value": gcs_uri, "mime_type": mime_type}

    def _analyze_multiple_media_with_gemini(
        self,
        client,
        all_media_sources: list,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Analyze multiple media items using Gemini model and return the response."""
        self._log(f"🤖 Analyzing {len(all_media_sources)} media item(s) with Gemini model: {model}")
        self._log("🔍 Starting _analyze_multiple_media_with_gemini method")

        # Prepare the contents list
        contents = []

        # Add the prompt text first
        if prompt:
            contents.append(prompt)
        else:
            contents.append("Please analyze and describe all the provided media content in detail.")

            # Add each media source to contents
        for i, media_source in enumerate(all_media_sources):
            self._log(f"📁 Adding media item {i + 1}: {media_source['type']}")

            if media_source["type"] == "url":
                # Public URL - use directly
                contents.append({"file_data": {"file_uri": media_source["value"]}})
            else:  # GCS URI
                # GCS URI - use directly with MIME type
                contents.append(
                    {
                        "file_data": {
                            "file_uri": media_source["value"],
                            "mime_type": media_source.get("mime_type", "application/octet-stream"),
                        }
                    }
                )

        # Generate content with all media
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
            )

            if response.candidates and response.candidates[0].content:
                return response.candidates[0].content.parts[0].text
            raise ValueError("No response generated from Gemini model")

        except Exception as e:
            # Handle "Service agents are being provisioned" error
            if "FAILED_PRECONDITION" in str(e) and "Service agents are being provisioned" in str(e):
                self._log("⚠️ Service agents are being provisioned. Retrying with inline data...")
                # For now, just re-raise the error since we don't have the original bytes
                raise
            # Re-raise other errors
            raise

    def process(self) -> AsyncResult:
        if not GOOGLE_INSTALLED:
            self._log(
                "ERROR: Required Google libraries are not installed. Please add 'google-cloud-aiplatform', 'google-generativeai' to your library's dependencies."
            )
            return
            yield  # unreachable but makes the function a generator

        # Get input values
        media_artifacts = self.get_parameter_value("media")
        prompt = self.get_parameter_value("prompt")
        model = self.get_parameter_value("model")
        temperature = self.get_parameter_value("temperature")
        max_tokens = self.get_parameter_value("max_tokens")
        location = self.get_parameter_value("location")

        # Validate inputs
        if not media_artifacts:
            self._log("ERROR: Media artifact is a required input.")
            return

        # Ensure media_artifacts is a list
        if not isinstance(media_artifacts, list):
            media_artifacts = [media_artifacts]

        self._log(f"📁 Processing {len(media_artifacts)} media item(s)...")

        try:
            final_project_id = None
            credentials = None

            # Try service account file first
            service_account_file = GriptapeNodes.SecretsManager().get_secret(f"{self.SERVICE_ACCOUNT_FILE_PATH}")
            if service_account_file and Path(service_account_file).exists():
                self._log("🔑 Using service account file for authentication.")
                try:
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_file
                    final_project_id = self._get_project_id(service_account_file)
                    credentials = service_account.Credentials.from_service_account_file(service_account_file)
                    self._log(f"✅ Service account authentication successful for project: {final_project_id}")
                except Exception as e:
                    self._log(f"❌ Service account file authentication failed: {e}")
                    raise
            else:
                # Fall back to individual credentials from settings
                self._log("🔑 Service account file not found, using individual credentials from settings.")
                project_id = GriptapeNodes.SecretsManager().get_secret(f"{self.PROJECT_ID}")
                credentials_json = GriptapeNodes.SecretsManager().get_secret(f"{self.CREDENTIALS_JSON}")

                if not project_id:
                    msg = "❌ GOOGLE_CLOUD_PROJECT_ID must be set in library settings when not using a service account file."
                    raise ValueError(msg)

                if credentials_json:
                    try:
                        import json

                        cred_dict = json.loads(credentials_json)
                        credentials = service_account.Credentials.from_service_account_info(cred_dict)
                        self._log("✅ JSON credentials authentication successful.")
                    except Exception as e:
                        self._log(f"❌ JSON credentials authentication failed: {e}")
                        raise
                else:
                    self._log("🔑 Using Application Default Credentials (e.g., gcloud auth).")

                final_project_id = project_id

            self._log(f"Project ID: {final_project_id}")
            self._log("Initializing Vertex AI...")
            aiplatform.init(project=final_project_id, location=location, credentials=credentials)

            self._log("Initializing Generative AI Client...")
            client = genai.Client(vertexai=True, project=final_project_id, location=location)

            # Log client configuration for debugging
            self._log(f"🔍 Client configured with project: {final_project_id}")
            self._log(f"🔍 Client configured with location: {location}")
            self._log("🔍 Client configured with vertexai: True")

            # Process all media artifacts and collect their data
            all_media_sources = []

            for i, media_artifact in enumerate(media_artifacts):
                self._log(f"📁 Processing media item {i + 1}/{len(media_artifacts)}...")

                media_source = self._process_media_artifact(media_artifact, final_project_id, credentials, location)
                all_media_sources.append(media_source)

            # Set the media type output (simplified)
            self.parameter_output_values["media_type"] = "mixed media"

            # Analyze all media with Gemini
            output = self._analyze_multiple_media_with_gemini(
                client,
                all_media_sources,
                prompt,
                model,
                temperature,
                max_tokens,
            )

            # Set the outputs
            self.parameter_output_values["output"] = output
            self.parameter_output_values["media_count"] = len(media_artifacts)

            self._log("✅ Media analysis completed successfully!")
            self._log(f"📝 Response: {output[:200]}...")
            self._log(f"📊 Processed {len(media_artifacts)} media item(s)")

        except ValueError as e:
            self._log(f"❌ CONFIGURATION ERROR: {e}")
            self._log("💡 Please set up Google Cloud credentials in the library settings:")
            self._log("   - GOOGLE_SERVICE_ACCOUNT_FILE_PATH (path to service account JSON)")
            self._log("   - OR GOOGLE_CLOUD_PROJECT_ID + GOOGLE_APPLICATION_CREDENTIALS_JSON")
            self._log("💡 Also ensure the Generative Language API is enabled for your project:")
            self._log(
                "   - Visit: https://console.developers.google.com/apis/api/generativelanguage.googleapis.com/overview"
            )
            self._log("   - Select your project and enable the API")
        except Exception as e:
            self._log(f"❌ An unexpected error occurred: {e}")
            import traceback

            self._log(traceback.format_exc())
