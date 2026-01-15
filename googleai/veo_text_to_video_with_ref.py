import logging
import time
from typing import Any, ClassVar

from griptape.artifacts import ImageArtifact, ImageUrlArtifact, VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.seed_parameter import SeedParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.events.os_events import ExistingFilePolicy
from griptape_nodes.traits.options import Options

# Attempt to import Google libraries
try:
    from google import genai
    from google.cloud import aiplatform, storage
    from google.genai.types import GenerateVideosConfig, Image, VideoGenerationReferenceImage

    GOOGLE_INSTALLED = True
except ImportError:
    GOOGLE_INSTALLED = False

from googleai_utils import GoogleAuthHelper, detect_image_mime_from_bytes

logger = logging.getLogger("griptape_nodes_library_googleai")

# Only models that support reference images
MODELS = [
    "veo-3.1-generate-preview",
    "veo-2.0-generate-exp",
]

# Model capabilities configuration
MODEL_CAPABILITIES = {
    "veo-3.1-generate-preview": {
        "max_reference_images": 3,
        "supports_reference_type_choice": False,  # Only supports "asset"
        "resolution_choices": ["720p", "1080p", "4k"],
        "duration_choices": [8],
        "duration_default": 8,
        "version": "veo3",
    },
    "veo-2.0-generate-exp": {
        "max_reference_images": 1,  # Only first reference image
        "supports_reference_type_choice": True,  # Can choose "asset" or "style"
        "resolution_choices": [],  # Veo 2.0 doesn't support resolution parameter
        "duration_choices": [5, 6, 7, 8],
        "duration_default": 8,
        "version": "veo2",
    },
}


class VeoTextToVideoWithRef(ControlNode):
    # Class-level cache for GCS clients
    _gcs_client_cache: ClassVar[dict[str, Any]] = {}

    # Service constants for configuration
    SERVICE = "GoogleAI"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category = "Google AI"
        self.description = "Generates videos using Google's Veo model with reference images. Only includes models that support reference images."

        # Main Parameters
        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="The text prompt for video generation.",
                multiline=True,
                placeholder_text="Describe the video you want to generate",
                allow_output=False,
            )
        )

        self.add_parameter(
            ParameterString(
                name="negative_prompt",
                tooltip="Optional: Text describing what you want to discourage the model from generating.",
                multiline=False,
                placeholder_text="Optional negative prompt",
                allow_output=False,
            )
        )

        self.add_parameter(
            ParameterString(
                name="model",
                tooltip="The Veo model to use for generation (all models support reference images).",
                default_value=MODELS[0],
                traits=[Options(choices=MODELS)],
                allow_output=False,
            )
        )

        # Reference type (for veo-2.0-generate-exp)
        self.add_parameter(
            ParameterString(
                name="reference_type",
                tooltip="Type of reference image: 'asset' (up to 3 images for veo-3.1, 1 for veo-2.0) or 'style' (1 image only, veo-2.0 only).",
                default_value="asset",
                traits={Options(choices=["asset", "style"])},
                allow_output=False,
            )
        )

        # Reference images (always visible - all models support them)
        self.add_parameter(
            ParameterImage(
                name="reference_image_1",
                tooltip="First reference image (required).",
                allowed_modes={ParameterMode.INPUT},
            )
        )

        self.add_parameter(
            ParameterImage(
                name="reference_image_2",
                tooltip="Second reference image (optional, veo-3.1-generate-preview only).",
                allowed_modes={ParameterMode.INPUT},
            )
        )

        self.add_parameter(
            ParameterImage(
                name="reference_image_3",
                tooltip="Third reference image (optional, veo-3.1-generate-preview only).",
                allowed_modes={ParameterMode.INPUT},
            )
        )

        self.add_parameter(
            ParameterString(
                name="aspect_ratio",
                tooltip="Aspect ratio of the generated video.",
                default_value="16:9",
                traits={Options(choices=["16:9", "9:16"])},
                allow_output=False,
            )
        )

        self.add_parameter(
            ParameterString(
                name="resolution",
                tooltip="Resolution of the generated video. Veo 3.1: 720p/1080p/4k. Not supported by Veo 2.0.",
                default_value="720p",
                traits={Options(choices=["720p", "1080p", "4k"])},
                allow_output=False,
            )
        )

        # Seed parameter component
        self._seed_parameter = SeedParameter(self)
        self._seed_parameter.add_input_parameters()

        # Duration parameter (choices vary by model)
        default_model = self.get_parameter_value("model") or MODELS[0]
        default_capabilities = MODEL_CAPABILITIES.get(default_model, MODEL_CAPABILITIES[MODELS[0]])
        self.add_parameter(
            ParameterInt(
                name="duration",
                tooltip="Duration of the generated video in seconds. Veo 2.0: 5-8 seconds. Veo 3.1: 4, 6, or 8 seconds.",
                default_value=default_capabilities["duration_default"],
                traits={Options(choices=default_capabilities["duration_choices"])},
                allow_output=False,
            )
        )

        self.add_parameter(
            ParameterBool(
                name="generate_audio",
                tooltip="Generate audio for the video (Veo 3 models only).",
                default_value=False,
                allow_output=False,
            )
        )

        self.add_parameter(
            ParameterInt(
                name="number_of_videos",
                tooltip="Number of videos to generate (sampleCount).",
                default_value=1,
                traits={Options(choices=[1, 2, 3, 4])},
                allow_output=False,
            )
        )

        self.add_parameter(
            ParameterString(
                name="location",
                tooltip="Google Cloud location for the generation job.",
                default_value="us-central1",
                traits={
                    Options(
                        choices=["us-central1", "us-east1", "us-west1", "europe-west1", "europe-west4", "asia-east1"]
                    )
                },
                allow_output=False,
            )
        )

        # Output Parameters using grid specification
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

        logs_group.ui_options = {"collapsed": True}
        self.add_node_element(logs_group)

        # Initialize video output visibility based on default number of videos
        default_num_videos = self.get_parameter_value("number_of_videos") or 1
        self._update_video_output_visibility(default_num_videos)

        # Initialize parameter visibility based on default model
        default_model = self.get_parameter_value("model") or MODELS[0]
        self._update_reference_image_visibility_for_model(default_model)
        self._update_generate_audio_visibility_for_model(default_model)
        self._update_resolution_choices_for_model(default_model)

    def _update_reference_image_visibility_for_model(self, model: str) -> None:
        """Update reference image visibility based on the selected model."""
        if model not in MODEL_CAPABILITIES:
            return

        capabilities = MODEL_CAPABILITIES[model]
        max_refs = capabilities.get("max_reference_images", 0)

        # Always show first reference image (required)
        self.show_parameter_by_name("reference_image_1")

        # Show/hide additional reference images based on max count
        if max_refs >= 3:
            # veo-3.1-generate-preview: show all 3
            self.show_parameter_by_name("reference_image_2")
            self.show_parameter_by_name("reference_image_3")
        else:
            # veo-2.0-generate-exp: only show first (max_refs = 1)
            self.hide_parameter_by_name("reference_image_2")
            self.hide_parameter_by_name("reference_image_3")

    def _update_generate_audio_visibility_for_model(self, model: str) -> None:
        """Update generate_audio visibility based on the selected model (Veo 3 only)."""
        if model not in MODEL_CAPABILITIES:
            return

        capabilities = MODEL_CAPABILITIES[model]
        version = capabilities.get("version", "")

        # Only show generate_audio for Veo 3 models
        if version == "veo3":
            self.show_parameter_by_name("generate_audio")
        else:
            self.hide_parameter_by_name("generate_audio")

    def _update_duration_choices_for_model(self, model: str) -> None:
        """Update duration choices based on the selected model."""
        if model not in MODEL_CAPABILITIES:
            return

        capabilities = MODEL_CAPABILITIES[model]
        current_duration = self.get_parameter_value("duration")
        if current_duration in capabilities["duration_choices"]:
            self._update_option_choices("duration", capabilities["duration_choices"], current_duration)
        else:
            # Set to default if current value is not in new choices
            self._update_option_choices("duration", capabilities["duration_choices"], capabilities["duration_default"])

    def _update_resolution_choices_for_model(self, model: str) -> None:
        """Update resolution choices based on the selected model (Veo 3.1 supports 4k, Veo 2.0 doesn't support resolution)."""
        if model not in MODEL_CAPABILITIES:
            return

        capabilities = MODEL_CAPABILITIES[model]
        resolution_choices = capabilities.get("resolution_choices", [])

        if resolution_choices:
            current_resolution = self.get_parameter_value("resolution") or "720p"
            if current_resolution not in resolution_choices:
                current_resolution = resolution_choices[0]
            self._update_option_choices("resolution", resolution_choices, current_resolution)
            self.show_parameter_by_name("resolution")
        else:
            # Model doesn't support resolution parameter - hide it
            self.hide_parameter_by_name("resolution")

    def _update_video_output_visibility(self, num_videos: int) -> None:
        """Update video output parameter visibility based on number of videos."""
        # Always show video_1_1 (first video)
        self.show_parameter_by_name("video_1_1")

        # Show/hide additional videos based on count
        if num_videos >= 2:
            self.show_parameter_by_name("video_1_2")
        else:
            self.hide_parameter_by_name("video_1_2")

        if num_videos >= 3:
            self.show_parameter_by_name("video_2_1")
        else:
            self.hide_parameter_by_name("video_2_1")

        if num_videos >= 4:
            self.show_parameter_by_name("video_2_2")
        else:
            self.hide_parameter_by_name("video_2_2")

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Handle parameter value changes."""
        if parameter.name == "model":
            self._update_duration_choices_for_model(value)
            self._update_resolution_choices_for_model(value)
            self._update_reference_image_visibility_for_model(value)
            self._update_generate_audio_visibility_for_model(value)
        elif parameter.name == "number_of_videos":
            self._update_video_output_visibility(value)
        self._seed_parameter.after_value_set(parameter, value)
        return super().after_value_set(parameter, value)

    def _log(self, message: str):
        """Append a message to the logs output parameter."""
        logger.info(message)
        self.append_value_to_parameter("logs", message + "\n")

    def _reset_outputs(self) -> None:
        """Clear output parameters so stale values don't persist across re-adds/reruns."""
        try:
            self.parameter_output_values["logs"] = ""
        except Exception:
            # Be defensive if the base class changes how outputs are stored
            pass

    def _get_image_base64(self, image_artifact) -> tuple[str, str]:
        """Convert image artifact to base64 string and return base64 data and mime type."""
        self._log("🖼️ Converting image to base64...")

        import requests

        # Get image data based on artifact type
        if isinstance(image_artifact, ImageUrlArtifact):
            # Download image from URL
            self._log(f"📥 Downloading image from URL: {image_artifact.value}")
            response = requests.get(image_artifact.value, timeout=30)
            response.raise_for_status()
            image_data = response.content

            # Determine mime type from response headers
            content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
            # If MIME type is missing or generic, detect from bytes
            if not content_type or content_type == "application/octet-stream":
                mime_type = detect_image_mime_from_bytes(image_data) or "image/png"
            elif "png" in content_type:
                mime_type = "image/png"
            elif "webp" in content_type:
                mime_type = "image/webp"
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

            # Detect mime type from bytes
            mime_type = detect_image_mime_from_bytes(image_data) or "image/png"
        else:
            raise ValueError(f"Unsupported image artifact type: {type(image_artifact)}")

        # Convert to base64
        import base64

        base64_data = base64.b64encode(image_data).decode("utf-8")

        self._log(f"✅ Image converted to base64 ({len(base64_data)} characters)")

        return base64_data, mime_type

    def _get_gcs_client(self, project_id: str, credentials):
        """Get a cached or new GCS client."""
        if project_id in self._gcs_client_cache:
            return self._gcs_client_cache[project_id]
        client = storage.Client(project=project_id, credentials=credentials)
        self._gcs_client_cache[project_id] = client
        return client

    def _download_from_gcs(self, gcs_uri: str, project_id: str, credentials) -> bytes:
        """Download video from GCS URI and return bytes."""
        self._log(f"📥 Downloading from GCS URI: {gcs_uri}")

        if not gcs_uri.startswith("gs://"):
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")

        path_parts = gcs_uri[5:].split("/", 1)
        bucket_name = path_parts[0]
        blob_path = path_parts[1]

        storage_client = self._get_gcs_client(project_id, credentials)

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

            if hasattr(operation, "error") and operation.error:
                self._log(f"❌ Operation has error: {operation.error}")
                return

            if not operation.response:
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
                    filename = f"veo_video_{int(time.time())}_{i + 1}.mp4"
                    self._log(f"Saving video bytes to static storage as {filename}...")

                    static_files_manager = GriptapeNodes.StaticFilesManager()
                    url = static_files_manager.save_static_file(video_bytes, filename, ExistingFilePolicy.CREATE_NEW)

                    url_artifact = VideoUrlArtifact(value=url, name=filename)
                    video_artifacts.append(url_artifact)
                    self._log(f"✅ Video {i + 1} saved. URL: {url}")
                else:
                    self._log(f"❌ Could not retrieve video data for video {i + 1}.")

            if video_artifacts:
                # Set the entire list of videos at once for grid display
                self.parameter_output_values["video_artifacts"] = video_artifacts
                # Proactively publish update to refresh UI
                try:
                    self.publish_update_to_parameter("video_artifacts", video_artifacts)
                except Exception:
                    pass

                # Assign each video to its individual grid position output
                for i, video in enumerate(video_artifacts):
                    row = (i // 2) + 1  # Row: 1, 1, 2, 2
                    col = (i % 2) + 1  # Col: 1, 2, 1, 2
                    param_name = f"video_{row}_{col}"
                    self.parameter_output_values[param_name] = video
                    self._log(f"📍 Assigned video {i + 1} to grid position {param_name}")
                    # Publish each individual param to help UI binders update
                    try:
                        self.publish_update_to_parameter(param_name, video)
                    except Exception:
                        pass

                self._log("\n🎉 SUCCESS! All videos processed.")
            else:
                self._log("\n❌ No videos were successfully saved.")
        except Exception as e:
            self._log(f"❌ An unexpected error occurred during polling: {e}")
            import traceback

            self._log(traceback.format_exc())
            raise

    def process(self) -> AsyncResult:
        # Clear outputs at the start of each run
        self._reset_outputs()

        if not GOOGLE_INSTALLED:
            self._log(
                "ERROR: Required Google libraries are not installed. Please add 'google-auth', 'google-cloud-aiplatform', 'google-cloud-storage', 'google-genai' to your library's dependencies."
            )
            return
            yield  # unreachable but makes the function a generator

        # Get input values
        prompt = self.get_parameter_value("prompt")
        negative_prompt = self.get_parameter_value("negative_prompt")
        model = self.get_parameter_value("model")
        aspect_ratio = self.get_parameter_value("aspect_ratio")
        resolution = self.get_parameter_value("resolution")
        self._seed_parameter.preprocess()
        seed = self._seed_parameter.get_seed()
        duration = self.get_parameter_value("duration")
        generate_audio = self.get_parameter_value("generate_audio")
        num_videos = self.get_parameter_value("number_of_videos")
        location = self.get_parameter_value("location")
        reference_image_1 = self.get_parameter_value("reference_image_1")
        reference_image_2 = self.get_parameter_value("reference_image_2")
        reference_image_3 = self.get_parameter_value("reference_image_3")
        reference_type = self.get_parameter_value("reference_type") or "asset"

        # Validate inputs
        if not prompt:
            self._log("ERROR: Prompt is a required input.")
            return

        if not reference_image_1:
            self._log("ERROR: At least one reference image is required.")
            return

        try:
            # Use GoogleAuthHelper for authentication
            credentials, final_project_id = GoogleAuthHelper.get_credentials_and_project(
                GriptapeNodes.SecretsManager(), log_func=self._log
            )

            self._log(f"Project ID: {final_project_id}")
            self._log("Initializing Vertex AI...")
            aiplatform.init(project=final_project_id, location=location, credentials=credentials)

            self._log("Initializing Generative AI Client...")
            client = genai.Client(
                vertexai=True, project=final_project_id, location=location, credentials=credentials
            )

            # Process reference images
            reference_images = []
            capabilities = MODEL_CAPABILITIES.get(model, {})
            max_refs = capabilities.get("max_reference_images", 0)
            supports_type_choice = capabilities.get("supports_reference_type_choice", False)

            # Collect reference images based on model capabilities
            ref_image_list = [reference_image_1]
            if max_refs >= 3 and reference_image_2:
                ref_image_list.append(reference_image_2)
            if max_refs >= 3 and reference_image_3:
                ref_image_list.append(reference_image_3)

            # Determine reference type
            if supports_type_choice:
                # veo-2.0-generate-exp: use user's choice
                ref_type = reference_type.lower()
                if ref_type not in ["asset", "style"]:
                    ref_type = "asset"  # Default to asset
                # For style, only 1 image is allowed
                if ref_type == "style" and len(ref_image_list) > 1:
                    self._log("⚠️ Warning: Style reference type only supports 1 image. Using first image only.")
                    ref_image_list = ref_image_list[:1]
            else:
                # veo-3.1: only supports "asset"
                ref_type = "asset"

            self._log(f"🖼️ Processing {len(ref_image_list)} reference image(s) with type '{ref_type}'...")

            for ref_img in ref_image_list:
                if not ref_img:
                    continue

                # Handle dict input
                if isinstance(ref_img, dict):
                    try:
                        if ref_img.get("type") == "ImageUrlArtifact":
                            ref_img = ImageUrlArtifact(value=ref_img.get("value"))
                        elif "value" in ref_img:
                            ref_img = ImageUrlArtifact(value=ref_img["value"])
                    except Exception as e:
                        self._log(f"⚠️ Failed to convert reference image dict: {e}")
                        continue

                try:
                    ref_base64, ref_mime = self._get_image_base64(ref_img)
                    reference_images.append(
                        VideoGenerationReferenceImage(
                            image=Image(
                                image_bytes=ref_base64,
                                mime_type=ref_mime,
                            ),
                            reference_type=ref_type,
                        )
                    )
                except Exception as e:
                    self._log(f"⚠️ Failed to process reference image: {e}")
                    continue

            if reference_images:
                self._log(f"✅ Processed {len(reference_images)} reference image(s) with type '{ref_type}'")
            else:
                self._log("❌ ERROR: Failed to process any reference images.")
                return

            self._log(f"🎬 Generating video for prompt: '{prompt}'")

            # Build config
            config_kwargs = {
                "aspect_ratio": aspect_ratio,
                "number_of_videos": num_videos,
                "reference_images": reference_images,
            }

            # Add resolution only for models that support it (Veo 3.1)
            if capabilities.get("resolution_choices") and resolution:
                config_kwargs["resolution"] = resolution

            # Add durationSeconds if provided
            if duration:
                config_kwargs["duration_seconds"] = duration

            # Add generateAudio if provided (Veo 3 models only)
            if generate_audio:
                config_kwargs["generate_audio"] = True

            # Add seed - SeedParameter handles randomization logic
            config_kwargs["seed"] = seed

            # Add negative prompt if provided - goes in config
            if negative_prompt:
                config_kwargs["negative_prompt"] = negative_prompt

            # Build API parameters
            api_params = {
                "model": model,
                "prompt": prompt,
                "config": GenerateVideosConfig(**config_kwargs),
            }

            operation = client.models.generate_videos(**api_params)

            self._log("⏳ Operation started! Waiting for completion...")

            # Use yield pattern for non-blocking execution
            yield lambda: self._poll_and_process_video_result(client, operation, final_project_id, credentials)

        except ValueError as e:
            self._log(f"❌ CONFIGURATION ERROR: {e}")
            self._log("💡 Please set up Google Cloud credentials in the library settings:")
            self._log("   - GOOGLE_WORKLOAD_IDENTITY_CONFIG_PATH (recommended, path to workload identity config)")
            self._log("   - OR GOOGLE_SERVICE_ACCOUNT_FILE_PATH (path to service account JSON)")
            self._log("   - OR GOOGLE_CLOUD_PROJECT_ID + GOOGLE_APPLICATION_CREDENTIALS_JSON")
        except Exception as e:
            self._log(f"❌ An unexpected error occurred: {e}")
            import traceback

            self._log(traceback.format_exc())
