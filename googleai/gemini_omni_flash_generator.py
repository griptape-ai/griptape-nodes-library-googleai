import base64
import logging
import time
from typing import Any

from googleai_utils import (
    GoogleAuthHelper,
    detect_image_mime_from_bytes,
    validate_and_maybe_shrink_image,
)
from griptape.artifacts import ImageArtifact, ImageUrlArtifact, VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

# Attempt to import Google libraries
try:
    from google import genai
    from google.cloud import aiplatform

    GOOGLE_INSTALLED = True
except ImportError:
    GOOGLE_INSTALLED = False

logger = logging.getLogger("griptape_nodes_library_googleai")

MODEL = "gemini-omni-flash-preview"

VERTEX_AI = "Vertex AI"
AI_STUDIO_API = "AI Studio API"

# The Interactions API is only served from the `global` Vertex location; regional
# endpoints (e.g. us-central1-aiplatform.googleapis.com) return 404 for it.
VERTEX_LOCATION = "global"

# Task values for the interactions video_config.
TASK_TEXT_TO_VIDEO = "text_to_video"
TASK_IMAGE_TO_VIDEO = "image_to_video"

# Terminal interaction statuses (the API also reports in_progress / requires_action).
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "incomplete"}

# Gemini Omni Flash accepts images up to a few MB; reuse a conservative 7 MB cap
# and the standard raster mime set the other Google nodes use.
MAX_IMAGE_BYTES = 7 * 1024 * 1024
ALLOWED_IMAGE_MIME = {"image/png", "image/jpeg", "image/webp"}

# Polling cadence for the background interaction.
POLL_INTERVAL_SECONDS = 10


class GeminiOmniFlashVideoGenerator(ControlNode):
    """Generate a video with Google's Gemini Omni Flash model via the native Google AI SDK.

    Gemini Omni Flash turns a text prompt (and an optional image) into a short, 720p
    video with audio using the Gemini Interactions API (client.interactions). The
    interaction runs in the background and is polled until it reaches a terminal state.

    Supports both auth surfaces: Vertex AI (default, service-account credentials) and
    the AI Studio API (GOOGLE_API_KEY). Google documents Omni on the AI Studio surface;
    Vertex availability is provider-dependent, so switch providers if one path rejects
    the model.
    """

    SERVICE = "GoogleAI"
    API_KEY = "GOOGLE_API_KEY"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "Google AI"
        self.description = "Generates videos using Google's Gemini Omni Flash model."

        # Auth provider selection
        self.add_parameter(
            ParameterString(
                name="api_provider",
                tooltip="Which Google surface to call. Vertex AI uses service-account credentials; AI Studio API uses GOOGLE_API_KEY.",
                default_value=VERTEX_AI,
                traits={Options(choices=[VERTEX_AI, AI_STUDIO_API])},
                allow_output=False,
            )
        )

        # Main inputs
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
            ParameterImage(
                name="image",
                tooltip="Optional input image. When provided, the model runs image-to-video.",
                allowed_modes={ParameterMode.INPUT},
                default_value=None,
                allow_output=False,
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

        # Output: single video artifact
        self.add_parameter(
            Parameter(
                name="video",
                type="VideoUrlArtifact",
                output_type="VideoUrlArtifact",
                tooltip="The generated video.",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        # Logs group (collapsed) — matches the Veo node.
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

        self._output_file = ProjectFileParameter(
            node=self, name="output_file", default_filename="gemini_omni_flash_video.mp4"
        )
        self._output_file.add_parameter()

    def _log(self, message: str) -> None:
        """Append a message to the logs output parameter."""
        logger.info(message)
        self.append_value_to_parameter("logs", message + "\n")

    def _reset_outputs(self) -> None:
        """Clear output parameters so stale values don't persist across re-adds/reruns."""
        try:
            self.parameter_output_values["logs"] = ""
            self.parameter_output_values["video"] = None
        except Exception:
            # Be defensive if the base class changes how outputs are stored.
            pass

    def _build_client(self, api_provider: str):
        """Construct a genai Client for the selected auth provider."""
        if api_provider == AI_STUDIO_API:
            api_key = GriptapeNodes.SecretsManager().get_secret(f"{self.API_KEY}")
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY must be set in library settings to use the AI Studio API. "
                    "Get your API key from https://aistudio.google.com/apikey"
                )
            self._log("🔑 Using Google AI Studio API key for authentication.")
            return genai.Client(api_key=api_key)

        # Vertex AI — interactions is only served from the global location.
        self._log("🔑 Using Vertex AI authentication.")
        credentials, project_id = GoogleAuthHelper.get_credentials_and_project(
            GriptapeNodes.SecretsManager(), log_func=self._log
        )
        self._log(f"Project ID: {project_id}")
        aiplatform.init(project=project_id, location=VERTEX_LOCATION, credentials=credentials)
        return genai.Client(vertexai=True, project=project_id, location=VERTEX_LOCATION, credentials=credentials)

    def _image_to_base64(self, art: Any) -> tuple[str, str]:
        """Return (base64_data, mime_type) for an image artifact input."""
        if isinstance(art, ImageArtifact):
            image_bytes = art.value
            mime = getattr(art, "mime_type", None)
            if not mime or mime == "application/octet-stream":
                mime = detect_image_mime_from_bytes(image_bytes) or "image/png"
        elif isinstance(art, ImageUrlArtifact):
            image_bytes = File(art.value).read_bytes()
            mime = detect_image_mime_from_bytes(image_bytes) or "image/png"
        else:
            raise TypeError(f"Unsupported image artifact type: {type(art)}")

        image_bytes, mime = validate_and_maybe_shrink_image(
            image_bytes=image_bytes,
            mime_type=mime,
            image_name=getattr(art, "name", "image"),
            allowed_mimes=ALLOWED_IMAGE_MIME,
            byte_limit=MAX_IMAGE_BYTES,
            log_func=self._log,
        )
        return base64.b64encode(image_bytes).decode("utf-8"), mime

    def process(self) -> AsyncResult:
        self._reset_outputs()

        if not GOOGLE_INSTALLED:
            self._log(
                "ERROR: Required Google libraries are not installed. Please add 'google-auth', "
                "'google-cloud-aiplatform', 'google-genai' to your library's dependencies."
            )
            return
            yield  # unreachable, but makes this a generator

        api_provider = self.get_parameter_value("api_provider") or VERTEX_AI
        prompt = self.get_parameter_value("prompt")
        aspect_ratio = self.get_parameter_value("aspect_ratio") or "16:9"
        image = self.get_parameter_value("image")

        if not prompt:
            self._log("ERROR: Prompt is a required input.")
            return

        try:
            client = self._build_client(api_provider)

            # Build the interactions `input`: a plain prompt for text-to-video, or a
            # list of content items (image + text) for image-to-video.
            if image:
                image_b64, mime_type = self._image_to_base64(image)
                model_input: Any = [
                    {"type": "image", "data": image_b64, "mime_type": mime_type},
                    {"type": "text", "text": prompt},
                ]
                task = TASK_IMAGE_TO_VIDEO
            else:
                model_input = prompt
                task = TASK_TEXT_TO_VIDEO

            self._log(f"🎬 Generating video for prompt: '{prompt}' (task: {task})")

            interaction = client.interactions.create(
                model=MODEL,
                input=model_input,
                background=True,
                response_format={"type": "video", "aspect_ratio": aspect_ratio},
                generation_config={"video_config": {"task": task}},
            )

            self._log(f"⏳ Interaction started (id: {interaction.id}). Waiting for completion...")
            yield lambda: self._poll_and_process(client, interaction.id)

        except ValueError as e:
            self._log(f"❌ CONFIGURATION ERROR: {e}")
        except Exception as e:
            self._log(f"❌ An unexpected error occurred: {e}")

    def _poll_and_process(self, client, interaction_id: str) -> None:
        """Poll the background interaction until terminal, then save the video."""
        try:
            interaction = client.interactions.get(interaction_id)
            while interaction.status not in TERMINAL_STATUSES:
                time.sleep(POLL_INTERVAL_SECONDS)
                interaction = client.interactions.get(interaction_id)
                self._log(f"⏳ Still generating... (status: {interaction.status})")

            if interaction.status != "completed":
                self._log(f"❌ Interaction ended with status '{interaction.status}'. No video produced.")
                return

            video_content = self._find_video_content(interaction)
            if video_content is None:
                self._log("❌ Interaction completed but contained no video output.")
                return

            video_bytes = self._video_content_to_bytes(client, video_content)
            if not video_bytes:
                self._log("❌ Could not retrieve video data from the interaction output.")
                return

            saved = self._output_file.build_file().write_bytes(video_bytes)
            url_artifact = VideoUrlArtifact(value=saved.location, name=saved.location)
            self.parameter_output_values["video"] = url_artifact
            self._log(f"✅ Saved video ({len(video_bytes)} bytes) to {saved.location}")

        except Exception as e:
            self._log(f"❌ An unexpected error occurred during polling: {e}")
            import traceback

            self._log(traceback.format_exc())
            raise

    @staticmethod
    def _find_video_content(interaction) -> Any:
        """Return the completed interaction's video content, or None.

        The SDK exposes the generated video via the `output_video` convenience
        accessor (`output_video.data`/`.uri`/`.mime_type`). Fall back to scanning
        the raw `steps[].content[]` for a video item if that accessor is absent.
        """
        output_video = getattr(interaction, "output_video", None)
        if output_video is not None:
            return output_video

        for step in getattr(interaction, "steps", None) or []:
            for content in getattr(step, "content", None) or []:
                if getattr(content, "type", None) == "video":
                    return content
        return None

    def _video_content_to_bytes(self, client, video_content) -> bytes | None:
        """Decode inline base64 video data, or download the file URI if that's what was returned."""
        data = getattr(video_content, "data", None)
        if data:
            return base64.b64decode(data)

        uri = getattr(video_content, "uri", None)
        if uri:
            self._log(f"📥 Downloading video from URI: {uri}")
            file_bytes = client.files.download(file=uri)
            return bytes(file_bytes) if file_bytes else None

        return None
