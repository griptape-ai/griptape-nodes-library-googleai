import logging
from typing import Any

from googleai_utils import (
    CREDENTIALS_HELP,
    GoogleAuthHelper,
    detect_image_mime_from_bytes,
    validate_and_maybe_shrink_image,
)
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

logger = logging.getLogger("griptape_nodes_library_googleai")

try:
    import io as _io

    from PIL import Image as PILImage

    PIL_INSTALLED = True
except ImportError:
    PIL_INSTALLED = False

try:
    from google import genai
    from google.genai import types

    GOOGLE_INSTALLED = True
except ImportError:
    GOOGLE_INSTALLED = False

MODELS = [
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-lite-image",
]
DEFAULT_MODEL = MODELS[0]

# Verified 2026-09-16 by generating at each size: the Lite model rejects every size except 1K
# with 400 INVALID_ARGUMENT, so the choices have to follow the selected model rather than being
# one shared list.
MODEL_IMAGE_SIZES: dict[str, list[str]] = {
    "gemini-3.1-flash-image": ["512", "1K", "2K", "4K"],
    "gemini-3.1-flash-lite-image": ["1K"],
}
DEFAULT_IMAGE_SIZES = MODEL_IMAGE_SIZES[DEFAULT_MODEL]

VERTEX_AI = "Vertex AI"
AI_STUDIO_API = "AI Studio API"


class NanaBanana2ImageGenerator(ControlNode):
    """Nano Banana 2 image generation node (Gemini 3.1 Flash).

    Supports both Vertex AI and Google AI Studio API:
    - Models: gemini-3.1-flash-image and gemini-3.1-flash-lite-image (same ids on both surfaces)
    - Supports up to 10 input images (≤ 7 MB each; png/jpeg/webp/heic/heif)
    - Uses genai.Client() SDK with response_modalities=['TEXT', 'IMAGE']
    - Supports 0.5K (512), 1K, 2K, and 4K resolution
    - Supports Google Image Search grounding
    - Returns generated images as ImageUrlArtifact
    """

    SERVICE = "GoogleAI"
    API_KEY = "GOOGLE_API_KEY"  # For Google AI Studio API

    # Model constraints: https://ai.google.dev/gemini-api/docs/image-generation
    MAX_PROMPT_IMAGES = 10
    MAX_IMAGE_BYTES = 7 * 1024 * 1024  # 7 MB
    ALLOWED_IMAGE_MIME = {"image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # ===== Core configuration =====
        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="User prompt for image generation.",
                multiline=True,
                placeholder_text="Enter prompt...",
                allow_output=True,
            )
        )

        self.add_parameter(
            Parameter(
                name="api_provider",
                type="str",
                tooltip="Choose API provider: Vertex AI (requires service account) or AI Studio API (requires API key).",
                default_value=VERTEX_AI,
                traits=[Options(choices=[AI_STUDIO_API, VERTEX_AI])],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="location",
                type="str",
                tooltip="Google Cloud location for Vertex AI (only used with Vertex AI provider).",
                default_value="global",
                traits=[Options(choices=["global", "us-central1", "europe-west1", "asia-southeast1"])],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # No Options trait here: ModelAccessComponent installs its own, plus the license
        # decoration and the legacy-value migration.
        model_parameter = ParameterString(
            name="model",
            tooltip=(
                "Which Nano Banana 2 model to use. The Lite model trades some quality for lower latency and cost."
            ),
            default_value=DEFAULT_MODEL,
            allow_output=False,
        )
        self.add_parameter(model_parameter)
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_parameter,
            model_choices=MODELS,
            default_model=DEFAULT_MODEL,
        )

        # ===== Reference Images =====
        self.add_parameter(
            ParameterList(
                name="reference_images",
                tooltip=f"Up to {self.MAX_PROMPT_IMAGES} reference images for style, context, or guidance (png/jpeg/webp/heic/heif, ≤ 7 MB each).",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="auto_image_resize",
                type="bool",
                tooltip="If disabled, raises an error when input images exceed the 7MB limit. If enabled, oversized images are best-effort scaled to fit within the 7MB limit.",
                default_value=True,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="use_google_search",
                type="bool",
                tooltip="Enable Google Search grounding to allow the model to search the web for up-to-date information.",
                default_value=False,
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="use_google_image_search",
                type="bool",
                tooltip="Enable Google Image Search grounding to allow the model to search Google Images for visual reference.",
                default_value=False,
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
            )
        )

        # ===== Image Configuration =====
        self.add_parameter(
            Parameter(
                name="aspect_ratio",
                type="str",
                tooltip="Aspect ratio for generated images.",
                default_value="16:9",
                traits=[
                    Options(
                        choices=[
                            "1:1",
                            "1:4",
                            "1:8",
                            "2:3",
                            "3:2",
                            "3:4",
                            "4:1",
                            "4:3",
                            "4:5",
                            "5:4",
                            "8:1",
                            "9:16",
                            "16:9",
                            "21:9",
                        ]
                    )
                ],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="image_size",
                type="str",
                tooltip="Resolution for generated images. The Lite model only produces 1K.",
                default_value="2K",
                traits=[Options(choices=DEFAULT_IMAGE_SIZES)],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # Temperature
        self.add_parameter(
            ParameterFloat(
                name="temperature",
                tooltip="Temperature for controlling generation randomness (0.0-2.0)",
                default_value=1.0,
                slider=True,
                min_val=0.0,
                max_val=2.0,
                step=0.1,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            ParameterFloat(
                name="top_p",
                tooltip="Top-p nucleus sampling (0.0–1.0).",
                default_value=0.95,
                slider=True,
                min_val=0.0,
                max_val=1.0,
                step=0.05,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # ===== Outputs =====
        self.add_parameter(
            Parameter(
                name="image",
                tooltip="First generated image",
                output_type="ImageUrlArtifact",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="images",
                tooltip="All generated images",
                output_type="list[ImageUrlArtifact]",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="text",
                tooltip="Text response from the model",
                output_type="str",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"multiline": True, "placeholder_text": "Generated text will appear here"},
                hide=True,
            )
        )

        # ===== Logs =====
        with ParameterGroup(name="Logs") as logs_group:
            Parameter(
                name="logs",
                type="str",
                tooltip="Processing logs.",
                ui_options={"multiline": True, "placeholder_text": "Logs"},
                allowed_modes={ParameterMode.OUTPUT},
            )
        self.add_node_element(logs_group)

        self._output_file = ProjectFileParameter(
            node=self, name="output_file", default_filename="gemini_3_1_flash_image.png"
        )
        self._output_file.add_parameter()

        # Ensure outputs are clean on (re)initialization
        self._reset_outputs()
        self._update_image_size_choices(self.get_parameter_value("model") or DEFAULT_MODEL)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "api_provider":
            if value == VERTEX_AI:
                self.show_parameter_by_name("location")
            else:
                self.hide_parameter_by_name("location")
        elif parameter.name == "model":
            self._update_image_size_choices(value)
        self._model_access.on_value_set(parameter, value)
        return super().after_value_set(parameter, value)

    def _update_image_size_choices(self, model: str) -> None:
        """Narrow the resolution choices to the ones the selected model accepts."""
        sizes = MODEL_IMAGE_SIZES.get(model)
        if sizes is None:
            return
        current = self.get_parameter_value("image_size")
        self._update_option_choices("image_size", sizes, current if current in sizes else sizes[-1])

    # ---------- Utilities ----------
    def _log(self, message: str):
        logger.info(message)
        self.append_value_to_parameter("logs", message + "\n")

    def _reset_outputs(self) -> None:
        """Clear every output so stale values don't persist across re-adds/reruns."""
        self._clear_image_outputs()
        self.parameter_output_values["logs"] = ""

    def _clear_image_outputs(self) -> None:
        """Clear the image outputs but keep the logs.

        The failure paths use this rather than `_reset_outputs`: the log is the only record of
        what went wrong, so emptying it on the way to raising would discard the explanation.
        """
        self.parameter_output_values["image"] = None
        self.parameter_output_values["images"] = []
        self.parameter_output_values["text"] = ""

    def _create_image_artifact(self, image_bytes: bytes) -> ImageUrlArtifact:
        saved = self._output_file.build_file().write_bytes(image_bytes)
        return ImageUrlArtifact(value=saved.location, name=saved.location)

    def _image_artifact_to_pil_image(
        self, art: Any, suggested_name: str = None, auto_image_resize: bool = True
    ) -> PILImage.Image:
        """Convert ImageArtifact or ImageUrlArtifact to PIL Image.

        Args:
            art: ImageArtifact or ImageUrlArtifact
            suggested_name: Optional name hint for the image (for logging/debugging)
            auto_image_resize: If False, fail when image exceeds 7 MB instead of auto-shrinking
        """
        if not PIL_INSTALLED:
            raise RuntimeError("Pillow is required to process images. Install 'Pillow' to enable.")

        # Get raw bytes and mime type from artifact
        if isinstance(art, ImageArtifact):
            image_bytes = art.value
            mime = getattr(art, "mime_type", None)
            # If MIME type is missing or generic, detect from bytes
            if not mime or mime == "application/octet-stream":
                mime = detect_image_mime_from_bytes(image_bytes) or "image/png"
        elif isinstance(art, ImageUrlArtifact):
            image_bytes = File(art.value).read_bytes()
            mime = detect_image_mime_from_bytes(image_bytes) or "image/png"
        else:
            raise TypeError(f"Unsupported image artifact type: {type(art)}")

        # Validate MIME type and size, shrink if needed
        img_name = suggested_name or getattr(art, "name", "image")
        image_bytes, mime = validate_and_maybe_shrink_image(
            image_bytes=image_bytes,
            mime_type=mime,
            image_name=img_name,
            allowed_mimes=self.ALLOWED_IMAGE_MIME,
            byte_limit=self.MAX_IMAGE_BYTES,
            auto_image_resize=auto_image_resize,
            log_func=self._log,
        )

        # Convert to PIL Image
        pil_img = PILImage.open(_io.BytesIO(image_bytes))
        if suggested_name:
            pil_img.filename = suggested_name
        return pil_img

    def _process_images(self, input_images: list, auto_image_resize: bool = True) -> list[PILImage.Image]:
        """Process and validate input images, return PIL Images.

        Args:
            input_images: List of ImageArtifact or ImageUrlArtifact
            auto_image_resize: If False, fail when image exceeds 7 MB instead of auto-shrinking

        Returns:
            List of PIL Images (max 10)
        """
        pil_images = []

        # Normalize to list
        images = input_images or []
        if not isinstance(images, list):
            images = [images]

        # Process images (max 10)
        for img_idx, img_art in enumerate(images[: self.MAX_PROMPT_IMAGES]):
            try:
                suggested_name = f"image{img_idx + 1}"
                pil_img = self._image_artifact_to_pil_image(
                    img_art, suggested_name=suggested_name, auto_image_resize=auto_image_resize
                )
                pil_images.append(pil_img)
            except Exception as e:
                img_name = getattr(img_art, "name", f"image_{img_idx + 1}")
                self._log(f"⚠️ Skipping image '{img_name}' due to error: {e}")

        if len(images) > self.MAX_PROMPT_IMAGES:
            self._log(f"ℹ️ Only the first {self.MAX_PROMPT_IMAGES} images are used.")

        return pil_images

    # ---------- Core generation ----------
    def _generate_and_process(
        self,
        client,
        model,
        prompt,
        input_images,
        aspect_ratio,
        image_size,
        use_google_search,
        use_google_image_search,
        temperature,
        top_p,
        auto_image_resize,
    ):
        """Generate image using Gemini 3.1 Flash and process response."""
        # Process input images
        pil_images = self._process_images(input_images, auto_image_resize=auto_image_resize)

        self._log(f"📸 Processing {len(pil_images)} input image(s)...")

        # Build contents list: prompt + images
        contents = [prompt] if prompt else []
        contents.extend(pil_images)

        # Build config - matching notebook pattern exactly
        # ImageConfig is available in google-genai >= 1.40.0
        config_kwargs = {
            "response_modalities": ["TEXT", "IMAGE"],
            "temperature": temperature,
            "top_p": top_p,
        }

        # Add grounding tools if enabled
        if use_google_search or use_google_image_search:
            try:
                if use_google_search and use_google_image_search:
                    search_config = {"webSearch": {}, "imageSearch": {}}
                    self._log("🔍 Google Search and Google Image Search grounding enabled.")
                elif use_google_image_search:
                    search_config = {"imageSearch": {}}
                    self._log("🖼️ Google Image Search grounding enabled.")
                else:
                    search_config = {}
                    self._log("🔍 Google Search grounding enabled.")
                google_search_tool = types.Tool(google_search=search_config)
                config_kwargs["tools"] = [google_search_tool]
            except (AttributeError, TypeError) as e:
                self._log(f"⚠️ Could not enable grounding tools: {e}")
                self._log("💡 Search grounding may require a specific API version or configuration")

        # Aspect ratio and output size travel in ImageConfig rather than on the top-level
        # config, where they would be rejected as unknown fields.
        config_kwargs["image_config"] = types.ImageConfig(
            aspect_ratio=aspect_ratio,
            image_size=image_size,
        )

        config = types.GenerateContentConfig(**config_kwargs)

        self._log("🎛️ Generation parameters:")
        self._log(f"  • Model: {model}")
        self._log(f"  • Aspect ratio: {aspect_ratio}")
        self._log(f"  • Image size: {image_size}")
        self._log(f"  • Temperature: {temperature}")
        self._log(f"  • Top-p: {top_p}")
        self._log(f"  • Google Search: {'Enabled' if use_google_search else 'Disabled'}")
        self._log(f"  • Google Image Search: {'Enabled' if use_google_image_search else 'Disabled'}")
        self._log(f"  • Input images: {len(pil_images)}")

        # Make API call - matching notebook pattern
        self._log("🧠 Calling Gemini 3.1 Flash generate_content API...")
        self._log("⏳ This may take 30-60 seconds or longer, especially with multiple reference images...")

        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            self._log("✅ API call completed successfully.")
        except Exception as e:
            error_msg = str(e)
            self._log(f"❌ API call failed: {error_msg}")
            raise

        self._log("📦 Processing response...")

        # Process response parts
        all_images = []
        text_parts = []

        # Get parts from the correct location in the response structure
        parts_to_process = None

        # Try direct parts attribute (older API structure)
        if hasattr(response, "parts") and response.parts:
            parts_to_process = response.parts
            self._log("📋 Found parts directly on response")
        # Try candidates structure (newer API structure)
        elif hasattr(response, "candidates") and response.candidates:
            if len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    if hasattr(candidate.content, "parts") and candidate.content.parts:
                        parts_to_process = candidate.content.parts
                        self._log("📋 Found parts in response.candidates[0].content.parts")

        if not parts_to_process:
            self._log("⚠️ Response has no parts to process.")
            return

        self._log(f"📋 Processing {len(parts_to_process)} response part(s)...")

        for idx, part in enumerate(parts_to_process):
            try:
                # Check if this is a "thought" part (internal reasoning, not final output)
                is_thought = getattr(part, "thought", False)
                thought_label = " (thought)" if is_thought else ""

                # Handle text parts
                if hasattr(part, "text") and part.text is not None:
                    # Skip thought parts or include them based on preference
                    if not is_thought:
                        text_parts.append(part.text)
                        self._log(f"📝 Part {idx + 1}: Text ({len(part.text)} chars){thought_label}")
                    else:
                        self._log(f"💭 Part {idx + 1}: Thought text ({len(part.text)} chars) - skipping")

                # Handle inline_data (Blob) - new structure
                elif hasattr(part, "inline_data") and part.inline_data:
                    blob = part.inline_data
                    image_bytes = blob.data
                    mime_type = getattr(blob, "mime_type", "image/png")
                    self._log(
                        f"🖼️ Part {idx + 1}: Image via inline_data ({len(image_bytes)} bytes, {mime_type}){thought_label}"
                    )

                    # Create artifact
                    art = self._create_image_artifact(image_bytes)
                    all_images.append(art)

                # Handle as_image() method - older structure
                elif hasattr(part, "as_image"):
                    try:
                        image = part.as_image()
                        if image:
                            image_bytes = image.image_bytes
                            mime_type = getattr(image, "mime_type", "image/png")
                            self._log(
                                f"🖼️ Part {idx + 1}: Image via as_image() ({len(image_bytes)} bytes, {mime_type}){thought_label}"
                            )

                            # Create artifact
                            art = self._create_image_artifact(image_bytes)
                            all_images.append(art)
                    except (AttributeError, ValueError) as e:
                        # A part that advertises as_image() but cannot produce bytes is not an
                        # image part; say so rather than dropping it without a trace.
                        self._log(f"ℹ️ Part {idx + 1}: as_image() yielded no image ({e}){thought_label}")
                else:
                    self._log(f"ℹ️ Part {idx + 1}: Unknown type (skipping){thought_label}")
            except Exception as e:
                self._log(f"⚠️ Error processing part {idx + 1}: {e}")

        # Set outputs
        if all_images:
            self.parameter_output_values["image"] = all_images[0]
            self.parameter_output_values["images"] = all_images
            self._log(f"🖼️ Saved {len(all_images)} image(s) to outputs.")
        else:
            self.parameter_output_values["image"] = None
            self.parameter_output_values["images"] = []
            self._log("ℹ️ No image outputs returned.")

        # Set text output
        combined_text = "\n".join(text_parts) if text_parts else ""
        self.parameter_output_values["text"] = combined_text
        if combined_text:
            self._log(f"📝 Text response saved ({len(combined_text)} characters).")

    # ---------- Node entrypoints ----------
    def validate_before_node_run(self) -> list[Exception] | None:
        """Reject a run that cannot possibly produce an image."""
        exceptions: list[Exception] = []

        if not GOOGLE_INSTALLED:
            exceptions.append(
                ImportError(f"{self.name}: 'google-genai' is not installed. Add it to this library's dependencies.")
            )
        if not PIL_INSTALLED:
            exceptions.append(
                ImportError(f"{self.name}: 'pillow' is not installed. Add it to this library's dependencies.")
            )
        if not self.get_parameter_value("prompt") and not self.get_parameter_value("reference_images"):
            exceptions.append(ValueError(f"{self.name}: provide at least a prompt or a reference image."))

        return exceptions or None

    def process(self) -> AsyncResult[None]:
        self._model_access.raise_if_selection_denied()
        yield lambda: self._process()

    def _process(self):
        # Clear outputs at the start of each run
        self._reset_outputs()

        # Get input values
        api_provider = self.get_parameter_value("api_provider")
        prompt = self.get_parameter_value("prompt")
        location = self.get_parameter_value("location")
        aspect_ratio = self.get_parameter_value("aspect_ratio")
        image_size = self.get_parameter_value("image_size")
        use_google_search = self.get_parameter_value("use_google_search")
        use_google_image_search = self.get_parameter_value("use_google_image_search")
        temperature = self.get_parameter_value("temperature")
        top_p = self.get_parameter_value("top_p")

        reference_images = self.get_parameter_value("reference_images") or []
        auto_image_resize = self.get_parameter_value("auto_image_resize")

        # Normalize to list
        if not isinstance(reference_images, list):
            reference_images = [reference_images]

        # Model ids are the same on both surfaces.
        model = self.get_parameter_value("model") or DEFAULT_MODEL

        self._log(f"📡 Using API provider: {api_provider}")
        self._log(f"🤖 Model: {model}")

        # Only the client construction counts as an auth failure; a ValueError raised later in the
        # run is not a credentials problem and must not be reported as one.
        try:
            if api_provider == "AI Studio API":
                # Use Google AI Studio API
                api_key = GriptapeNodes.SecretsManager().get_secret(f"{self.API_KEY}")
                if not api_key:
                    raise ValueError(
                        "❌ GOOGLE_API_KEY must be set in library settings to use AI Studio API. "
                        "Get your API key from https://aistudio.google.com/apikey"
                    )
                self._log("🔑 Using Google AI Studio API key for authentication.")
                client = genai.Client(api_key=api_key)
            else:  # Vertex AI
                # Use Vertex AI authentication
                self._log("🔑 Using Vertex AI authentication.")

                # Use GoogleAuthHelper for authentication
                credentials, project_id = GoogleAuthHelper.get_credentials_and_project(
                    GriptapeNodes.SecretsManager(), log_func=self._log
                )

                self._log(f"Project ID: {project_id}")

                self._log("Initializing Generative AI Client (Vertex AI)...")
                client = genai.Client(vertexai=True, project=project_id, location=location, credentials=credentials)

        except ValueError as e:
            self._clear_image_outputs()
            self._log(f"❌ Configuration error: {e}")
            msg = (
                f"{self.name}: could not authenticate to Google. {e} For the AI Studio API set "
                f"GOOGLE_API_KEY (from https://aistudio.google.com/apikey). For Vertex AI, {CREDENTIALS_HELP}"
            )
            raise RuntimeError(msg) from e

        try:
            self._log("🚀 Starting Gemini 3.1 Flash image generation...")
            self._generate_and_process(
                client=client,
                model=model,
                prompt=prompt,
                input_images=reference_images,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                use_google_search=use_google_search,
                use_google_image_search=use_google_image_search,
                temperature=temperature,
                top_p=top_p,
                auto_image_resize=auto_image_resize,
            )

        except Exception as e:
            self._clear_image_outputs()
            self._log(f"❌ Image generation failed: {e}")
            msg = f"{self.name}: Gemini image generation failed. {e}"
            raise RuntimeError(msg) from e
