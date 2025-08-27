import json
import os
import time

import requests
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes_library.video.video_url_artifact import VideoUrlArtifact

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

# Attempt to import Google libraries
try:
    from google import genai
    from google.cloud import aiplatform, storage
    from google.genai.types import GenerateVideosConfig, Image
    from google.oauth2 import service_account

    GOOGLE_INSTALLED = True
except ImportError:
    GOOGLE_INSTALLED = False


class VeoImageToVideoGenerator(ControlNode):
    # Service constants for configuration
    SERVICE = "GoogleAI"
    SERVICE_ACCOUNT_FILE_PATH = "GOOGLE_SERVICE_ACCOUNT_FILE_PATH"
    PROJECT_ID = "GOOGLE_CLOUD_PROJECT_ID"
    CREDENTIALS_JSON = "GOOGLE_APPLICATION_CREDENTIALS_JSON"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category = "Google AI"
        self.description = "Generates videos from an image input using Google's Veo model."

        # Main Parameters
        self.add_parameter(
            Parameter(
                name="image",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                type="ImageArtifact",
                tooltip="The input image for video generation.",
                allowed_modes={ParameterMode.INPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="prompt",
                type="str",
                tooltip="Optional: The text prompt for video generation to guide the animation.",
                ui_options={"multiline": True, "placeholder_text": "Optional text prompt to guide the animation"},
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="model",
                type="str",
                tooltip="The Veo model to use for generation.",
                default_value="veo-3.0-generate-001",
                traits={
                    Options(
                        choices=[
                            "veo-3.0-generate-001",
                            "veo-3.0-fast-generate-001",
                            "veo-3.0-generate-preview",
                            "veo-3.0-fast-generate-preview",
                        ]
                    )
                },
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="negative_prompt",
                type="str",
                tooltip="Optional: Text describing what you want to discourage the model from generating.",
                ui_options={"multiline": True, "placeholder_text": "Optional negative prompt"},
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="seed",
                type="int",
                tooltip="Optional: Seed number for deterministic generation (0-4294967295).",
                default_value=0,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="number_of_videos",
                type="int",
                tooltip="Number of videos to generate (sampleCount).",
                default_value=1,
                traits=[Options(choices=[1, 2, 3, 4])],
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="aspect_ratio",
                type="str",
                tooltip="Aspect ratio of the generated video. Note: 9:16 is not supported by veo-3.0-generate-preview.",
                default_value="16:9",
                traits=[Options(choices=["16:9", "9:16"])],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="resolution",
                type="str",
                tooltip="Resolution of the generated video.",
                default_value="720p",
                traits=[Options(choices=["720p", "1080p"])],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
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
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # Output Parameters using your EXACT grid specification
        grid_param = Parameter(
            name="video_artifacts",
            type="list",
            default_value=[],
            output_type="list[VideoUrlArtifact]",
            tooltip="Generated video artifacts (up to 4 videos)",
            ui_options={"display": "grid", "columns": 2},
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(grid_param)

        # Individual video output parameters for grid positions
        # Always add all 4, but hide 2-4 by default (shown when number_of_videos > 1)
        self.add_parameter(
            Parameter(
                name="video_1_1",
                type="VideoUrlArtifact",
                output_type="VideoUrlArtifact",
                tooltip="Video at grid position [1,1]",
                ui_options={"hide_property": True},
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="video_1_2",
                type="VideoUrlArtifact",
                output_type="VideoUrlArtifact",
                tooltip="Video at grid position [1,2]",
                ui_options={"hide_property": True, "hide_when": {"number_of_videos": [1]}},
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="video_2_1",
                type="VideoUrlArtifact",
                output_type="VideoUrlArtifact",
                tooltip="Video at grid position [2,1]",
                ui_options={"hide_property": True, "hide_when": {"number_of_videos": [1, 2]}},
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="video_2_2",
                type="VideoUrlArtifact",
                output_type="VideoUrlArtifact",
                tooltip="Video at grid position [2,2]",
                ui_options={"hide_property": True, "hide_when": {"number_of_videos": [1, 2, 3]}},
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        # Logs Group
        with ParameterGroup(name="Logs") as logs_group:
            Parameter(
                name="logs",
                type="str",
                tooltip="Logs from the video generation process.",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"multiline": True, "placeholder_text": "Logs"},
            )

        logs_group.ui_options = {"hide": True}
        self.add_node_element(logs_group)

    def _log(self, message: str):
        """Append a message to the logs output parameter."""
        print(message)
        self.append_value_to_parameter("logs", message + "\n")

    def _get_project_id(self, service_account_file: str) -> str:
        """Read the project_id from the service account JSON file."""
        if not os.path.exists(service_account_file):
            raise FileNotFoundError(f"Service account file not found: {service_account_file}")

        with open(service_account_file) as f:
            service_account_info = json.load(f)

        project_id = service_account_info.get("project_id")
        if not project_id:
            raise ValueError("No 'project_id' found in the service account file.")

        return project_id

    def _get_image_base64(self, image_artifact) -> tuple[str, str]:
        """Convert image artifact to base64 string and return base64 data and mime type."""
        self._log("🖼️ Converting image to base64...")

        # Get image data based on artifact type
        if isinstance(image_artifact, ImageUrlArtifact):
            # Download image from URL
            self._log(f"📥 Downloading image from URL: {image_artifact.value}")
            response = requests.get(image_artifact.value)
            response.raise_for_status()
            image_data = response.content

            # Determine mime type from URL or response headers
            content_type = response.headers.get("content-type", "image/jpeg")
            if "png" in content_type.lower():
                mime_type = "image/png"
            else:
                mime_type = "image/jpeg"

        elif isinstance(image_artifact, ImageArtifact):
            # Handle ImageArtifact
            if hasattr(image_artifact, "value") and hasattr(image_artifact.value, "read"):
                # If it's a file-like object
                image_data = image_artifact.value.read()
            elif hasattr(image_artifact, "base64"):
                # If it's already base64 encoded, return it directly
                return image_artifact.base64, "image/png"
            else:
                # Try to get the raw value
                image_data = image_artifact.value

            mime_type = "image/png"  # Default for ImageArtifact
        else:
            raise ValueError(f"Unsupported image artifact type: {type(image_artifact)}")

        # Convert to base64
        import base64

        base64_data = base64.b64encode(image_data).decode("utf-8")

        self._log(f"✅ Image converted to base64 ({len(base64_data)} characters)")

        return base64_data, mime_type

    def _download_from_gcs(self, gcs_uri: str, project_id: str, credentials) -> bytes:
        """Download video from GCS URI and return bytes."""
        self._log(f"📥 Downloading from GCS URI: {gcs_uri}")

        if not gcs_uri.startswith("gs://"):
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")

        path_parts = gcs_uri[5:].split("/", 1)
        bucket_name = path_parts[0]
        blob_path = path_parts[1]

        storage_client = storage.Client(project=project_id, credentials=credentials)

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        return blob.download_as_bytes()

    def _poll_and_process_video_result(self, client, operation, final_project_id, credentials) -> None:
        """Poll for video generation completion and process results - called via yield."""
        try:
            # Poll for completion
            while not operation.done:
                time.sleep(15)
                operation = client.operations.get(operation)
                self._log("⏳ Still generating...")

            self._log("✅ Video generation completed!")

            # Check for errors first
            if hasattr(operation, "error") and operation.error:
                error_details = operation.error
                self._log(f"❌ Video generation failed with error: {error_details}")

                # Provide user-friendly error explanation
                if isinstance(error_details, dict) and error_details.get("code") == 13:
                    self._log("🔄 This is a temporary Google API internal error. Please try again in a few minutes.")
                    self._log(
                        "💡 Tip: You can also try changing the location parameter to a different region (e.g., 'us-east1')."
                    )
                return

            if not hasattr(operation, "response") or not operation.response:
                self._log("❌ Video generation completed but no response found.")
                return

            video_artifacts = []
            # Fix: videos are in operation.response, not operation.result
            generated_videos = operation.response.generated_videos if operation.response else None

            # Check for content filtering
            if operation.response and hasattr(operation.response, "rai_media_filtered_count"):
                filtered_count = operation.response.rai_media_filtered_count
                if filtered_count > 0:
                    self._log(f"🚫 Content Filter: {filtered_count} video(s) were filtered by Google's content policy.")
                    if (
                        hasattr(operation.response, "rai_media_filtered_reasons")
                        and operation.response.rai_media_filtered_reasons
                    ):
                        for reason in operation.response.rai_media_filtered_reasons:
                            self._log(f"   Reason: {reason}")
                    self._log("💡 Tip: Try rephrasing your prompt to avoid violent, sexual, or harmful content.")
                    return

            if not generated_videos:
                self._log("❌ No videos found in the response.")
                return

            self._log(f"🎯 Generated {len(generated_videos)} video(s)")

            for i, video in enumerate(generated_videos):
                self._log(f"Processing video {i + 1}...")
                video_bytes = None

                # Check for direct video bytes
                if hasattr(video.video, "video_bytes") and video.video.video_bytes:
                    self._log(f"💾 Video {i + 1} returned as direct bytes.")
                    video_bytes = video.video.video_bytes
                # Fallback to downloading from GCS URI
                elif hasattr(video.video, "uri") and video.video.uri:
                    self._log(f"📹 Video {i + 1} has GCS URI. Downloading...")
                    video_bytes = self._download_from_gcs(video.video.uri, final_project_id, credentials)

                if video_bytes:
                    filename = f"veo_image_to_video_{int(time.time())}_{i + 1}.mp4"
                    self._log(f"Saving video bytes to static storage as {filename}...")

                    static_files_manager = GriptapeNodes.StaticFilesManager()
                    url = static_files_manager.save_static_file(video_bytes, filename)

                    url_artifact = VideoUrlArtifact(value=url, name=filename)
                    video_artifacts.append(url_artifact)
                    self._log(f"✅ Video {i + 1} saved. URL: {url}")
                else:
                    self._log(f"❌ Could not retrieve video data for video {i + 1}.")

            if video_artifacts:
                # Set the entire list of videos at once for grid display
                self.parameter_output_values["video_artifacts"] = video_artifacts

                # Assign each video to its individual grid position output
                for i, video in enumerate(video_artifacts):
                    row = (i // 2) + 1  # Row: 1, 1, 2, 2
                    col = (i % 2) + 1  # Col: 1, 2, 1, 2
                    param_name = f"video_{row}_{col}"
                    self.parameter_output_values[param_name] = video
                    self._log(f"📍 Assigned video {i + 1} to grid position {param_name}")

                self._log("\n🎉 SUCCESS! All videos processed.")
            else:
                self._log("\n❌ No videos were successfully saved.")
        except Exception as e:
            self._log(f"❌ An unexpected error occurred during polling: {e}")
            import traceback

            self._log(traceback.format_exc())
            raise

    def process(self) -> AsyncResult:
        if not GOOGLE_INSTALLED:
            self._log(
                "ERROR: Required Google libraries are not installed. Please add 'google-cloud-aiplatform', 'google-generativeai', 'google-cloud-storage' to your library's dependencies."
            )
            return
            yield  # unreachable but makes the function a generator

        # Get input values
        image_artifact = self.get_parameter_value("image")
        prompt = self.get_parameter_value("prompt")
        negative_prompt = self.get_parameter_value("negative_prompt")
        model = self.get_parameter_value("model")
        num_videos = self.get_parameter_value("number_of_videos")
        aspect_ratio = self.get_parameter_value("aspect_ratio")
        resolution = self.get_parameter_value("resolution")

        seed = self.get_parameter_value("seed")
        location = self.get_parameter_value("location")

        # Validate inputs
        if not image_artifact:
            self._log("ERROR: Image is a required input.")
            return

        # Validate aspect ratio for specific models
        if model == "veo-3.0-generate-preview" and aspect_ratio == "9:16":
            self._log("ERROR: 9:16 aspect ratio is not supported by veo-3.0-generate-preview model.")
            return

        try:
            final_project_id = None
            credentials = None

            # Try service account file first
            service_account_file = self.get_config_value(service=self.SERVICE, value=self.SERVICE_ACCOUNT_FILE_PATH)

            if service_account_file and os.path.exists(service_account_file):
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
                project_id = self.get_config_value(service=self.SERVICE, value=self.PROJECT_ID)
                credentials_json = self.get_config_value(service=self.SERVICE, value=self.CREDENTIALS_JSON)

                if not project_id:
                    raise ValueError(
                        "❌ GOOGLE_CLOUD_PROJECT_ID must be set in library settings when not using a service account file."
                    )

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

            # Convert image to base64
            base64_data, mime_type = self._get_image_base64(image_artifact)

            self._log(f"🎬 Generating video from image with prompt: '{prompt or 'No prompt provided'}'")

            # Build the API call parameters
            api_params = {
                "model": model,
                "image": Image(
                    image_bytes=base64_data,
                    mime_type=mime_type,
                ),
                "config": GenerateVideosConfig(
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    number_of_videos=num_videos,
                ),
            }

            # Add prompt if provided
            if prompt:
                api_params["prompt"] = prompt

            # Add negative prompt if provided
            if negative_prompt:
                api_params["negative_prompt"] = negative_prompt

            # Add seed if provided (non-zero)
            if seed and seed > 0:
                api_params["seed"] = seed

            operation = client.models.generate_videos(**api_params)

            self._log("⏳ Operation started! Waiting for completion...")

            # Use yield pattern for non-blocking execution
            yield lambda: self._poll_and_process_video_result(client, operation, final_project_id, credentials)

        except ValueError as e:
            self._log(f"❌ CONFIGURATION ERROR: {e}")
            self._log("💡 Please set up Google Cloud credentials in the library settings:")
            self._log("   - GOOGLE_SERVICE_ACCOUNT_FILE_PATH (path to service account JSON)")
            self._log("   - OR GOOGLE_CLOUD_PROJECT_ID + GOOGLE_APPLICATION_CREDENTIALS_JSON")
        except Exception as e:
            self._log(f"❌ An unexpected error occurred: {e}")
            import traceback

            self._log(traceback.format_exc())
