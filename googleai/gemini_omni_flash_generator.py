import base64
import logging
import time
from typing import Any

from googleai_utils import (
    CREDENTIALS_HELP,
    GoogleAuthHelper,
    detect_image_mime_from_bytes,
    validate_and_maybe_shrink_image,
)
from griptape.artifacts import ImageArtifact, ImageUrlArtifact, VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, BaseNode, ControlNode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

# Attempt to import Google libraries
try:
    from google import genai

    GOOGLE_INSTALLED = True
except ImportError:
    GOOGLE_INSTALLED = False

logger = logging.getLogger("griptape_nodes_library_googleai")

MODELS = [
    "gemini-omni-1.1-flash-preview",
]
DEFAULT_MODEL = MODELS[0]

# Google shuts the original Omni preview down on 2026-09-30 and names Omni 1.1 as its
# replacement, so a workflow saved against it is migrated on load rather than left to 404.
# https://ai.google.dev/gemini-api/docs/deprecations
DEPRECATED_MODELS = {
    "gemini-omni-flash-preview": DEFAULT_MODEL,
}

VERTEX_AI = "Vertex AI"
AI_STUDIO_API = "AI Studio API"

# The Interactions API is only served from the `global` Vertex location; regional
# endpoints (e.g. us-central1-aiplatform.googleapis.com) return 404 for it.
VERTEX_LOCATION = "global"

# Task values for the interactions video_config.
TASK_TEXT_TO_VIDEO = "text_to_video"
TASK_IMAGE_TO_VIDEO = "image_to_video"
TASK_REFERENCE_TO_VIDEO = "reference_to_video"

# Terminal interaction statuses (the API also reports in_progress / requires_action).
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "incomplete"}

# Gemini Omni Flash accepts images up to a few MB; reuse a conservative 7 MB cap
# and the standard raster mime set the other Google nodes use.
MAX_IMAGE_BYTES = 7 * 1024 * 1024
ALLOWED_IMAGE_MIME = {"image/png", "image/jpeg", "image/webp"}

# Polling cadence for the background interaction.
POLL_INTERVAL_SECONDS = 10


class GeminiOmniFlashVideoGenerator(ControlNode):
    """Generate a video with Google's Gemini Omni models via the native Google AI SDK.

    Gemini Omni turns a text prompt (and optionally a starting frame or a set of reference
    images) into a short, 720p video with audio using the Gemini Interactions API
    (client.interactions). The interaction runs in the background and is polled until it
    reaches a terminal state. Reference roles are assigned by prompt tags such as
    <IMAGE_REF_0>; audio references are not supported by these models.

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

        # Tracked from the connection hooks; there is no public API to query a parameter's
        # incoming connections, and the deprecated `image` stays visible while one exists.
        self._legacy_image_connected = False

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

        # No Options trait here: ModelAccessComponent installs its own, plus the license
        # decoration and the legacy-value migration.
        model_parameter = ParameterString(
            name="model",
            tooltip="The Gemini Omni model to use for generation.",
            default_value=DEFAULT_MODEL,
            allow_output=False,
        )
        self.add_parameter(model_parameter)
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_parameter,
            model_choices=MODELS,
            default_model=DEFAULT_MODEL,
            deprecated_values=DEPRECATED_MODELS,
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
            ParameterList(
                name="reference_images",
                input_types=["ImageUrlArtifact", "ImageArtifact"],
                default_value=[],
                tooltip=(
                    "Optional reference images. When any are provided the model runs "
                    "reference-to-video and 'image' is ignored. Refer to them in the prompt as "
                    "<IMAGE_REF_0>, <IMAGE_REF_1>, and so on (zero-indexed) to control how each "
                    "one is used. Audio references are not supported by this model."
                ),
                allowed_modes={ParameterMode.INPUT},
                ui_options={"display_name": "reference images", "expander": True, "hide_property": True},
            )
        )

        # Superseded by reference_images. Kept so workflows saved against it still load (renaming
        # a parameter breaks both its saved value and its saved connection on load), but hidden
        # unless a workflow actually uses it. See _update_legacy_image_visibility.
        self.add_parameter(
            ParameterImage(
                name="image",
                tooltip="Deprecated: use 'reference images' instead. When set, the model runs image-to-video.",
                allowed_modes={ParameterMode.INPUT},
                default_value=None,
                allow_output=False,
                ui_options={"display_name": "image (deprecated)"},
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

        self._update_legacy_image_visibility()

    def after_incoming_connection(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        if target_parameter.name == "image":
            self._legacy_image_connected = True
            self._update_legacy_image_visibility()
        return super().after_incoming_connection(source_node, source_parameter, target_parameter)

    def after_incoming_connection_removed(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        if target_parameter.name == "image":
            self._legacy_image_connected = False
            self._update_legacy_image_visibility()
        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "image":
            self._update_legacy_image_visibility()
        self._model_access.on_value_set(parameter, value)
        return super().after_value_set(parameter, value)

    def _update_legacy_image_visibility(self) -> None:
        """Show the deprecated `image` parameter only while a workflow still uses it.

        New graphs never see it; graphs saved before `reference_images` existed keep working and
        can still see what they are wired to.
        """
        in_use = bool(self.get_parameter_value("image")) or self._legacy_image_connected
        if in_use:
            self.show_parameter_by_name("image")
        else:
            self.hide_parameter_by_name("image")

    def _log(self, message: str) -> None:
        """Append a message to the logs output parameter."""
        logger.info(message)
        self.append_value_to_parameter("logs", message + "\n")

    def _reset_outputs(self) -> None:
        """Clear output parameters so stale values don't persist across re-adds/reruns."""
        self.parameter_output_values["logs"] = ""
        self.parameter_output_values["video"] = None

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

    def validate_before_node_run(self) -> list[Exception] | None:
        """Reject a run that cannot possibly produce a video."""
        exceptions: list[Exception] = []

        if not GOOGLE_INSTALLED:
            exceptions.append(
                ImportError(
                    f"{self.name}: the Google libraries are not installed. Add 'google-auth' and "
                    "'google-genai' to this library's dependencies."
                )
            )
        if not self.get_parameter_value("prompt"):
            exceptions.append(ValueError(f"{self.name}: a prompt is required."))

        return exceptions or None

    def process(self) -> AsyncResult:
        self._reset_outputs()
        self._model_access.raise_if_selection_denied()

        api_provider = self.get_parameter_value("api_provider") or VERTEX_AI
        model = self.get_parameter_value("model") or DEFAULT_MODEL
        prompt = self.get_parameter_value("prompt")
        aspect_ratio = self.get_parameter_value("aspect_ratio") or "16:9"
        image = self.get_parameter_value("image")
        reference_images = self.get_parameter_list_value("reference_images")

        # Only client construction counts as an auth failure. `_image_to_base64` below raises
        # ValueError for an unsupported MIME type or an image that will not fit under the size
        # cap, and neither is a credentials problem.
        try:
            client = self._build_client(api_provider)
        except ValueError as e:
            self.parameter_output_values["video"] = None
            self._log(f"❌ Configuration error: {e}")
            msg = f"{self.name}: could not authenticate to Google. {e} {CREDENTIALS_HELP}"
            raise RuntimeError(msg) from e

        try:
            # Build the interactions `input`: a plain prompt for text-to-video, or a list of
            # content items (images + text). Reference images and a starting frame mean different
            # things to the model, so they select different tasks rather than combining.
            if reference_images:
                if image:
                    self._log(
                        "WARNING: Both 'image' and 'reference_images' were provided. Using "
                        "reference_images (reference-to-video) and ignoring 'image'."
                    )
                reference_items = []
                for reference in reference_images:
                    reference_b64, reference_mime = self._image_to_base64(reference)
                    reference_items.append({"type": "image", "data": reference_b64, "mime_type": reference_mime})
                model_input: Any = [*reference_items, {"type": "text", "text": prompt}]
                task = TASK_REFERENCE_TO_VIDEO
            elif image:
                image_b64, mime_type = self._image_to_base64(image)
                model_input = [
                    {"type": "image", "data": image_b64, "mime_type": mime_type},
                    {"type": "text", "text": prompt},
                ]
                task = TASK_IMAGE_TO_VIDEO
            else:
                model_input = prompt
                task = TASK_TEXT_TO_VIDEO

            self._log(f"🎬 Generating video for prompt: '{prompt}' (task: {task})")

            interaction = client.interactions.create(
                model=model,
                input=model_input,
                background=True,
                response_format={"type": "video", "aspect_ratio": aspect_ratio},
                generation_config={"video_config": {"task": task}},
            )

            self._log(f"⏳ Interaction started (id: {interaction.id}). Waiting for completion...")
            yield lambda: self._poll_and_process(client, interaction.id)

        except Exception as e:
            self.parameter_output_values["video"] = None
            self._log(f"❌ Video generation failed: {e}")
            msg = f"{self.name}: Gemini Omni video generation failed. {e}"
            raise RuntimeError(msg) from e

    def _poll_and_process(self, client, interaction_id: str) -> None:
        """Poll the background interaction until terminal, then save the video."""
        try:
            interaction = client.interactions.get(interaction_id)
            while interaction.status not in TERMINAL_STATUSES:
                time.sleep(POLL_INTERVAL_SECONDS)
                interaction = client.interactions.get(interaction_id)
                self._log(f"⏳ Still generating... (status: {interaction.status})")

            if interaction.status != "completed":
                msg = f"Gemini Omni ended with status '{interaction.status}' and produced no video."
                raise RuntimeError(msg)

            video_content = self._find_video_content(interaction)
            if video_content is None:
                msg = "Gemini Omni reported completion but returned no video."
                raise RuntimeError(msg)

            video_bytes = self._video_content_to_bytes(client, video_content)
            if not video_bytes:
                msg = "Gemini Omni returned a video reference whose data could not be retrieved."
                raise RuntimeError(msg)

            saved = self._output_file.build_file().write_bytes(video_bytes)
            url_artifact = VideoUrlArtifact(value=saved.location, name=saved.location)
            self.parameter_output_values["video"] = url_artifact
            self._log(f"✅ Saved video ({len(video_bytes)} bytes) to {saved.location}")

        except Exception as e:
            self.parameter_output_values["video"] = None
            self._log(f"❌ Failed while waiting for the video: {e}")
            msg = f"{self.name}: Gemini Omni video generation failed while polling. {e}"
            raise RuntimeError(msg) from e

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
