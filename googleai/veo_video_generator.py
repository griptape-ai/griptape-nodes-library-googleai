import logging
import time
from typing import Any, ClassVar

from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.seed_parameter import SeedParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.events.os_events import ExistingFilePolicy
from griptape_nodes.traits.options import Options

# Attempt to import Google libraries
try:
    from google import genai
    from google.cloud import aiplatform, storage
    from google.genai.types import GenerateVideosConfig

    GOOGLE_INSTALLED = True
except ImportError:
    GOOGLE_INSTALLED = False

from googleai_utils import GoogleAuthHelper

logger = logging.getLogger("griptape_nodes_library_googleai")

MODELS = [
    "veo-3.1-generate-preview",
    "veo-3.1-fast-generate-preview",
    "veo-3.0-generate-001",
    "veo-3.0-fast-generate-001",
    "veo-3.0-generate-preview",
    "veo-3.0-fast-generate-preview",
]

# Model capabilities configuration
# Maps model names to their supported features
MODEL_CAPABILITIES = {
    "veo-3.1-generate-preview": {
        "duration_choices": [8],
        "duration_default": 8,
        "version": "veo3",
    },
    "veo-3.1-fast-generate-preview": {
        "duration_choices": [4, 6, 8],
        "duration_default": 8,
        "version": "veo3",
    },
    "veo-3.0-generate-001": {
        "duration_choices": [4, 6, 8],
        "duration_default": 8,
        "version": "veo3",
    },
    "veo-3.0-fast-generate-001": {
        "duration_choices": [4, 6, 8],
        "duration_default": 8,
        "version": "veo3",
    },
    "veo-3.0-generate-preview": {
        "duration_choices": [4, 6, 8],
        "duration_default": 8,
        "version": "veo3",
    },
    "veo-3.0-fast-generate-preview": {
        "duration_choices": [4, 6, 8],
        "duration_default": 8,
        "version": "veo3",
    },
}


class VeoVideoGenerator(ControlNode):
    # Class-level cache for GCS clients
    _gcs_client_cache: ClassVar[dict[str, Any]] = {}

    # Service constants for configuration
    SERVICE = "GoogleAI"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category = "Google AI"
        self.description = "Generates videos using Google's Veo model."

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
                tooltip="The Veo model to use for generation.",
                default_value=MODELS[0],
                traits=[Options(choices=MODELS)],
                allow_output=False,
            )
        )

        self.add_parameter(
            ParameterString(
                name="aspect_ratio",
                tooltip="Aspect ratio of the generated video. Note: 9:16 is not supported by veo-3.0-generate-preview.",
                default_value="16:9",
                traits={Options(choices=["16:9", "9:16"])},
                allow_output=False,
            )
        )

        self.add_parameter(
            ParameterString(
                name="resolution",
                tooltip="Resolution of the generated video.",
                default_value="720p",
                traits={Options(choices=["720p", "1080p"])},
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
                tooltip="Duration of the generated video in seconds. Veo 2.0: 5-8 seconds. Veo 3.0: 4, 6, or 8 seconds.",
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

        logs_group.ui_options = {"collapsed": True}
        self.add_node_element(logs_group)

        # Initialize duration choices based on default model
        default_model = self.get_parameter_value("model") or MODELS[0]
        self._update_duration_choices_for_model(default_model)

        # Initialize video output visibility based on default number of videos
        default_num_videos = self.get_parameter_value("number_of_videos") or 1
        self._update_video_output_visibility(default_num_videos)

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

        # Validate inputs
        if not prompt:
            self._log("ERROR: Prompt is a required input.")
            return

        # Validate aspect ratio for specific models
        if model == "veo-3.0-generate-preview" and aspect_ratio == "9:16":
            self._log("ERROR: 9:16 aspect ratio is not supported by veo-3.0-generate-preview model.")
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

            self._log(f"🎬 Generating video for prompt: '{prompt}'")

            # Build config
            config_kwargs = {
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "number_of_videos": num_videos,
            }

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
