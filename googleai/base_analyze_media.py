import base64
import hashlib
import json
import logging
import urllib.parse
from pathlib import Path
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes  # type: ignore[reportMissingImports]
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

# Attempt to import Google libraries
try:
    from google import genai
    from google.cloud import storage
    from google.genai import types

    GOOGLE_INSTALLED = True
except ImportError:
    GOOGLE_INSTALLED = False

from googleai_utils import credentials_or_raise
from griptape_nodes.files.file import File

logger = logging.getLogger("griptape_nodes_library_googleai")

MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
    # Shuts down 2027-05-07 with gemini-3.5-flash-lite as its replacement. Still offered because
    # that is far off; move it into DEPRECATED_MODELS as the date approaches.
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]
DEFAULT_MODEL = "gemini-3.6-flash"

# Verified 2026-09-16: every Gemini 3 model answers 404 from a regional endpoint and only serves
# from `global`, while the 2.5 models serve from both. A regional default would therefore make
# most of the list above unreachable, so `global` leads the location choices.
GLOBAL_LOCATION = "global"
GLOBAL_ONLY_MODELS = frozenset(MODELS) - {"gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"}

# Google shut the Gemini 2.0 models down on 2026-06-01, so a workflow saved against one is
# migrated on load to the successor Google names rather than left to 404 at run time.
# https://ai.google.dev/gemini-api/docs/deprecations
DEPRECATED_MODELS = {
    "gemini-2.0-flash": "gemini-3.6-flash",
    "gemini-2.0-flash-lite": "gemini-3.5-flash-lite",
}


class BaseAnalyzeMedia(ControlNode):
    # Service constants for configuration
    SERVICE = "GoogleAI"
    CLOUD_BUCKET_NAME = "GOOGLE_CLOUD_BUCKET_NAME"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "Media Analysis/Google AI"
        self.description = "Analyzes images, videos, or audio and answers questions about the media content using Google's Gemini model."

        # Main Parameters
        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="The prompt/question to ask about the media content.",
                multiline=True,
                placeholder_text="What would you like to know about this media?",
                allow_output=False,
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

        # No Options trait here: ModelAccessComponent installs its own, plus the license
        # decoration and the legacy-value migration.
        model_parameter = ParameterString(
            name="model",
            tooltip="The Gemini model to use for analysis.",
            default_value=DEFAULT_MODEL,
            allowed_modes={ParameterMode.PROPERTY},
        )
        self.add_parameter(model_parameter)
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_parameter,
            model_choices=MODELS,
            default_model=DEFAULT_MODEL,
            deprecated_values=DEPRECATED_MODELS,
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
                tooltip=(
                    "Maximum number of tokens in the response. The reasoning these models do before "
                    "answering is charged against this budget, so a low value can end a run before "
                    "any text is produced."
                ),
                # Sized for media analysis with reasoning charged against the same budget.
                default_value=8192,
                traits=[Options(choices=[1024, 2048, 4096, 8192, 16384, 32768])],
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={"hide": True},
            )
        )

        self.add_parameter(
            Parameter(
                name="location",
                type="str",
                tooltip=(
                    "Google Cloud location for the analysis. The Gemini 3 models are only served "
                    "from 'global'; the 2.5 models are served from any of these."
                ),
                default_value=GLOBAL_LOCATION,
                traits=[
                    Options(
                        choices=[
                            GLOBAL_LOCATION,
                            "us-central1",
                            "us-east1",
                            "us-west1",
                            "europe-west1",
                            "europe-west4",
                            "asia-east1",
                        ]
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
                ui_options={"multiline": True, "placeholder_text": "AI response will appear here (new)"},
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

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Handle parameter value changes."""
        self._model_access.on_value_set(parameter, value)
        return super().after_value_set(parameter, value)

    def _log(self, message: str) -> None:
        """Append a message to the logs output parameter."""
        self.append_value_to_parameter("logs", message + "\n")

    def _raise_file_not_found(self, file_path: str) -> None:
        """Raise FileNotFoundError with logging."""
        msg = f"Local file not found: {file_path}"
        logger.error(msg)
        self._log(msg)
        raise FileNotFoundError(msg)

    def _get_project_id(self, service_account_file: str) -> str:
        """Read the project_id from the service account JSON file."""
        if not Path(service_account_file).exists():
            msg = f"Service account file not found: {service_account_file}"
            logger.error(msg)
            self._log(msg)
            raise FileNotFoundError(msg)

        with Path(service_account_file).open(encoding="utf-8") as f:
            service_account_info = json.load(f)

        project_id = service_account_info.get("project_id")
        if not project_id:
            msg = "No 'project_id' found in the service account file."
            logger.error(msg)
            self._log(msg)
            raise ValueError(msg)

        return project_id

    def _upload_to_gcs(  # noqa: PLR0913
        self, media_data: bytes, filename: str, mime_type: str, project_id: str, credentials: Any, location: str
    ) -> str:
        """Upload media file to GCS bucket and return the GCS URI."""
        try:
            # Use the bucket name that matches your setup
            griptape_cloud_bucket_name = GriptapeNodes.ConfigManager().get_config_value(
                f"{self.SERVICE}.{self.CLOUD_BUCKET_NAME}"
            )
            if not griptape_cloud_bucket_name:
                msg = "GOOGLE_CLOUD_BUCKET_NAME is not set in the library settings. Using default bucket name 'griptape-nodes'."
                logger.warning(msg)
                bucket_name = "griptape-nodes"
            else:
                bucket_name = griptape_cloud_bucket_name

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

    def _get_media_source(self, media_artifact: Any) -> dict:
        """Get media source information for processing."""
        self._log("🔄 Processing media artifact...")

        # Check if it's a public URL (not localhost)
        if hasattr(media_artifact, "value") and isinstance(media_artifact.value, str):
            if "localhost" in media_artifact.value or "127.0.0.1" in media_artifact.value:
                return {"type": "localhost_url", "url": media_artifact.value}
            return {"type": "public_url", "url": media_artifact.value}

        # Direct artifact
        return {"type": "direct_artifact", "artifact": media_artifact}

    def _extract_bytes_from_artifact(self, media_artifact: Any) -> bytes:
        """Extract bytes from any media artifact."""
        if hasattr(media_artifact, "value") and hasattr(media_artifact.value, "read"):
            # File-like object
            return media_artifact.value.read()
        if hasattr(media_artifact, "base64"):
            # Base64 encoded
            return base64.b64decode(media_artifact.base64)
        # Direct bytes or other format
        return media_artifact.value

    def _generate_filename(self, media_artifact: Any, content_hash: str) -> str:
        """Generate filename with original name + content hash."""
        if hasattr(media_artifact, "value") and isinstance(media_artifact.value, str):
            # URL artifact - extract filename
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
        self,
        media_artifact: Any,
        project_id: str | None = None,
        credentials: Any = None,
        location: str | None = None,
    ) -> dict:
        """Process a single media artifact and return source info."""
        source = self._get_media_source(media_artifact)

        if source["type"] == "public_url":
            # Public URL - use directly with detected MIME type
            self._log(f"🌐 Using public URL: {source['url']}")
            # Extract filename from URL to determine MIME type
            url_path = source["url"].split("?")[0]  # Remove query params
            mime_type = self._get_mime_type(url_path)
            return {"type": "url", "value": source["url"], "mime_type": mime_type}

        if source["type"] == "localhost_url":
            # Localhost URL - read file and upload to GCS
            self._log(f"📁 Reading local file: {source['url']}")
            media_data = File(source["url"]).read_bytes()

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

            # Always include mime_type - the API requires it
            contents.append(
                {
                    "file_data": {
                        "file_uri": media_source["value"],
                        "mime_type": media_source.get("mime_type", "application/octet-stream"),
                    }
                }
            )

        # Generate content with all media
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=self._generation_config(temperature, max_tokens),
        )

        return self._read_response_text(response)

    @staticmethod
    def _read_response_text(response: Any) -> str:
        """Return the model's text, naming the real cause when there is none.

        `response.text` rather than the first part: a response routinely splits its text across
        several parts, and the thinking-capable models lead with parts that carry only a thought
        signature. The SDK accessor concatenates every text part, skips the thought parts, and
        answers None when there is no text at all, which is the signal the branches below need.
        """
        text = response.text
        if text:
            return text

        candidates = response.candidates or []
        if not candidates:
            msg = "Gemini returned no candidates for this request."
            raise ValueError(msg)

        finish_reason = getattr(candidates[0], "finish_reason", None)
        if finish_reason is not None and "MAX_TOKENS" in str(finish_reason):
            msg = (
                "Gemini stopped at the 'max_tokens' limit before producing any text. Raise "
                "'max_tokens' on this node: these models spend part of that budget on reasoning "
                "that never appears in the output."
            )
            raise ValueError(msg)

        msg = f"Gemini returned no text for this request (finish_reason: {finish_reason})."
        raise ValueError(msg)

    @staticmethod
    def _generation_config(temperature: float, max_tokens: int) -> Any:
        """Build the generate_content config.

        Lives on the base so subclasses can pass the node's sampling settings through without
        importing the SDK types themselves, which is guarded at the top of this module.
        """
        return types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens)

    def validate_before_node_run(self) -> list[Exception] | None:
        """Reject a run that cannot possibly produce an analysis."""
        exceptions: list[Exception] = []

        if not GOOGLE_INSTALLED:
            exceptions.append(
                ImportError(
                    f"{self.name}: the Google libraries are not installed. Add 'google-auth', "
                    "'google-cloud-storage' and 'google-genai' to this library's dependencies."
                )
            )
        if not self.get_parameter_value("media"):
            exceptions.append(ValueError(f"{self.name}: at least one media item is required."))

        return exceptions or None

    def _effective_location(self, model: str) -> str:
        """The location to call `model` at, which is not always the one the parameter holds.

        The Gemini 3 models are served only from `global`. Overriding here rather than rejecting
        the run is what lets a workflow saved against a retired 2.0 model keep working: the model
        is migrated on load, but the stored location is not, so it arrives pointing at a region
        its new model does not serve.
        """
        location = self.get_parameter_value("location")
        if model in GLOBAL_ONLY_MODELS and location != GLOBAL_LOCATION:
            self._log(f"ℹ️ '{model}' is only served from '{GLOBAL_LOCATION}'; using that instead of '{location}'.")
            return GLOBAL_LOCATION
        return location

    def process(self) -> AsyncResult:
        # The analysis is a single blocking SDK call that can run for minutes on a long video.
        # Handing it to the engine as a yielded callable keeps it off the event loop; doing the
        # work inline here would block every other node for its duration.
        self._model_access.raise_if_selection_denied()
        yield lambda: self._process()

    def _process(self) -> None:
        # Get input values
        media_artifacts = self.get_parameter_value("media")
        prompt = self.get_parameter_value("prompt")
        model = self.get_parameter_value("model")
        temperature = self.get_parameter_value("temperature")
        max_tokens = self.get_parameter_value("max_tokens")
        location = self._effective_location(model)

        # Ensure media_artifacts is a list
        if not isinstance(media_artifacts, list):
            media_artifacts = [media_artifacts]

        self._log(f"📁 Processing {len(media_artifacts)} media item(s)...")

        credentials, final_project_id = credentials_or_raise(
            self.name,
            log_func=self._log,
            on_failure=self._set_safe_defaults,
            extra_help=" Also confirm the Vertex AI API is enabled for the project.",
        )

        try:
            self._log(f"Project ID: {final_project_id}")
            self._log(f"Initializing Generative AI Client at '{location}'...")
            client = genai.Client(vertexai=True, project=final_project_id, location=location, credentials=credentials)

            # Process all media artifacts and collect their data
            all_media_sources = []

            for i, media_artifact in enumerate(media_artifacts):
                self._log(f"📁 Processing media item {i + 1}/{len(media_artifacts)}...")

                media_source = self._process_media_artifact(media_artifact, final_project_id, credentials, location)
                all_media_sources.append(media_source)

            self.parameter_output_values["media_type"] = self._describe_media_type(all_media_sources)

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

        except Exception as e:
            self._set_safe_defaults()
            self._log(f"❌ Media analysis failed: {e}")
            msg = f"{self.name}: media analysis failed. {e}"
            raise RuntimeError(msg) from e

    def _set_safe_defaults(self) -> None:
        """Clear the analysis outputs, keeping the logs that explain why.

        Downstream nodes read `output`; leaving a previous run's text in place would let a
        failure look like a success to everything wired after this node.
        """
        self.parameter_output_values["output"] = ""
        self.parameter_output_values["media_count"] = 0
        self.parameter_output_values["media_type"] = ""

    @staticmethod
    def _describe_media_type(media_sources: list[dict]) -> str:
        """Name the kind of media that was sent, for the `media_type` output.

        Reports the single category when every item agrees and "mixed media" only when they
        genuinely differ, so a run of three videos does not describe itself as mixed.
        """
        categories = {source.get("mime_type", "").split("/")[0] for source in media_sources if source.get("mime_type")}
        known = {category for category in categories if category in {"image", "video", "audio"}}
        if len(known) == 1:
            return known.pop()
        if known:
            return "mixed media"
        return "unknown"
