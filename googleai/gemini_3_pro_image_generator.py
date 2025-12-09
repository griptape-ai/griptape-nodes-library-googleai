import json
import logging
import os
import tempfile
import time
from typing import Any

from griptape.artifacts import ImageArtifact, ImageUrlArtifact

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

try:
    import io as _io

    from PIL import Image as PILImage

    PIL_INSTALLED = True
except Exception:
    PIL_INSTALLED = False

try:
    import requests

    REQUESTS_INSTALLED = True
except Exception:
    REQUESTS_INSTALLED = False

from googleai_utils import validate_and_maybe_shrink_image

logger = logging.getLogger("griptape_nodes_library_googleai")

try:
    from google import genai
    from google.cloud import aiplatform
    from google.genai import types
    from google.oauth2 import service_account

    GOOGLE_GENAI_VERSION = getattr(genai, "__version__", "unknown")

    # Try to import ImageConfig explicitly (available in google-genai >= 1.40.0)
    try:
        from google.genai.types import ImageConfig

        IMAGE_CONFIG_AVAILABLE = True
    except (ImportError, AttributeError) as e:
        logger.error(f"ImageConfig not available: {e}")
        IMAGE_CONFIG_AVAILABLE = False

    GOOGLE_INSTALLED = True
except ImportError as e:
    logger.error(f"Google libraries not installed: {e}")
    GOOGLE_INSTALLED = False
    IMAGE_CONFIG_AVAILABLE = False
    GOOGLE_GENAI_VERSION = "not installed"


VERTEX_AI = "Vertex AI"
AI_STUDIO_API = "AI Studio API"


class NanoBananaProImageGenerator(ControlNode):
    """Nano Banana Pro image generation node (Gemini 3 Pro).

    Supports both Vertex AI and Google AI Studio API:
    - Model: gemini-3-pro-image-preview (same model name for both APIs)
    - Supports up to 14 input images (≤ 7 MB each; png/jpeg/webp/heic/heif)
    - Uses genai.Client() SDK with response_modalities=['TEXT', 'IMAGE']
    - Supports 1K, 2K, and 4K resolution
    - Returns generated images as ImageUrlArtifact
    """

    SERVICE = "GoogleAI"
    SERVICE_ACCOUNT_FILE_PATH = "GOOGLE_SERVICE_ACCOUNT_FILE_PATH"
    PROJECT_ID = "GOOGLE_CLOUD_PROJECT_ID"
    CREDENTIALS_JSON = "GOOGLE_APPLICATION_CREDENTIALS_JSON"
    API_KEY = "GOOGLE_API_KEY"  # For Google AI Studio API

    # Model constraints: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/3-pro-image
    MAX_PROMPT_IMAGES = 14
    MAX_IMAGE_BYTES = 7 * 1024 * 1024  # 7 MB
    ALLOWED_IMAGE_MIME = {"image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # ===== Core configuration =====
        self.add_parameter(
            Parameter(
                name="prompt",
                input_types=["str"],
                type="str",
                output_type="str",
                tooltip="User prompt for image generation.",
                ui_options={"multiline": True, "placeholder_text": "Enter prompt..."},
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
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

        # ===== Reference Images =====
        self.add_parameter(
            ParameterList(
                name="reference_images",
                tooltip=f"Up to {self.MAX_PROMPT_IMAGES} reference images for style, context, or guidance (png/jpeg/webp/heic/heif, ≤ 7 MB each).",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # ===== Backwards Compatibility (hidden) =====
        # These parameters are kept for backwards compatibility with existing workflows
        self.add_parameter(
            ParameterList(
                name="object_images",
                tooltip="[Deprecated] Use reference_images instead.",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                allowed_modes={ParameterMode.INPUT},
                hide=True,
            )
        )
        self.add_parameter(
            ParameterList(
                name="human_images",
                tooltip="[Deprecated] Use reference_images instead.",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                allowed_modes={ParameterMode.INPUT},
                hide=True,
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

        # ===== Image Configuration =====
        self.add_parameter(
            Parameter(
                name="aspect_ratio",
                type="str",
                tooltip="Aspect ratio for generated images.",
                default_value="16:9",
                traits=[Options(choices=["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"])],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="image_size",
                type="str",
                tooltip="Resolution for generated images.",
                default_value="2K",
                traits=[Options(choices=["1K", "2K", "4K"])],
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
                ui_options={"multiline": True, "placeholder_text": "Generated text response..."},
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

        # Ensure outputs are clean on (re)initialization
        self._reset_outputs()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "api_provider":
            if value == VERTEX_AI:
                self.show_parameter_by_name("location")
            else:
                self.hide_parameter_by_name("location")
        return super().after_value_set(parameter, value)

    # ---------- Utilities ----------
    def _log(self, message: str):
        logger.info(message)
        self.append_value_to_parameter("logs", message + "\n")

    def _reset_outputs(self) -> None:
        """Clear output parameters so stale values don't persist across re-adds/reruns."""
        try:
            self.parameter_output_values["image"] = None
            self.parameter_output_values["images"] = []
            self.parameter_output_values["text"] = ""
            self.parameter_output_values["logs"] = ""
        except Exception:
            pass

    def _get_project_id(self, service_account_file: str) -> str:
        if not os.path.exists(service_account_file):
            raise FileNotFoundError(f"Service account file not found: {service_account_file}")
        with open(service_account_file) as f:
            return json.load(f).get("project_id")

    def _create_image_artifact(self, image_bytes: bytes, mime_type: str) -> ImageUrlArtifact:
        import hashlib

        timestamp = int(time.time() * 1000)
        content_hash = hashlib.md5(image_bytes).hexdigest()[:8]
        ext = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/webp": "webp",
        }.get(mime_type, "png")
        filename = f"Gemini3ProImage_{timestamp}_{content_hash}.{ext}"
        static_url = GriptapeNodes.StaticFilesManager().save_static_file(image_bytes, filename)
        return ImageUrlArtifact(value=static_url, name=f"gemini_3_pro_image_{timestamp}")

    def _image_artifact_to_pil_image(
        self, art: Any, suggested_name: str = None, auto_image_resize: bool = False
    ) -> PILImage.Image:
        """Convert ImageArtifact or ImageUrlArtifact to PIL Image.

        Args:
            art: ImageArtifact or ImageUrlArtifact
            suggested_name: Optional name hint for the image (for logging/debugging)
            auto_image_resize: If True, fail when image exceeds 7 MB instead of auto-shrinking
        """
        if not PIL_INSTALLED:
            raise RuntimeError("Pillow is required to process images. Install 'Pillow' to enable.")

        # Get raw bytes and mime type from artifact
        if isinstance(art, ImageArtifact):
            image_bytes = art.value
            mime = getattr(art, "mime_type", None) or "image/png"
        elif isinstance(art, ImageUrlArtifact):
            if not REQUESTS_INSTALLED:
                raise RuntimeError("`requests` is required to fetch ImageUrlArtifact URLs.")
            resp = requests.get(art.value, timeout=30)
            resp.raise_for_status()
            image_bytes = resp.content
            mime = resp.headers.get("Content-Type", "").split(";")[0].strip().lower() or "image/png"
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
            strict_size=auto_image_resize,
            log_func=self._log,
        )

        # Convert to PIL Image
        pil_img = PILImage.open(_io.BytesIO(image_bytes))
        if suggested_name:
            pil_img.filename = suggested_name
        return pil_img

    def _process_images(self, input_images: list, auto_image_resize: bool = False) -> list[PILImage.Image]:
        """Process and validate input images, return PIL Images.

        Args:
            input_images: List of ImageArtifact or ImageUrlArtifact
            auto_image_resize: If True, fail when image exceeds 7 MB instead of auto-shrinking

        Returns:
            List of PIL Images (max 14)
        """
        pil_images = []

        # Normalize to list
        images = input_images or []
        if not isinstance(images, list):
            images = [images]

        # Process images (max 14)
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
        temperature,
        top_p,
        auto_image_resize,
    ):
        """Generate image using Gemini 3 Pro and process response."""
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

        # Add Google Search tool if enabled
        if use_google_search:
            try:
                google_search_tool = types.Tool(google_search={})
                config_kwargs["tools"] = [google_search_tool]
                self._log("🔍 Google Search grounding enabled.")
            except (AttributeError, TypeError) as e:
                self._log(f"⚠️ Could not enable Google Search: {e}")
                self._log("💡 Google Search may require a specific API version or configuration")

        # Try to add image config if ImageConfig is available
        try:
            if IMAGE_CONFIG_AVAILABLE:
                # Use ImageConfig class (preferred method from notebook)
                image_config = types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                )
                config_kwargs["image_config"] = image_config
            # Try accessing ImageConfig from types namespace directly
            # (it might exist even if direct import failed)
            elif hasattr(types, "ImageConfig"):
                image_config = types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                )
                config_kwargs["image_config"] = image_config
            else:
                # ImageConfig not available - skip image config
                self._log("⚠️ ImageConfig not available - aspect_ratio and image_size will be ignored")
                self._log(f"💡 Current google-genai version: {GOOGLE_GENAI_VERSION}")
                self._log("💡 Ensure google-genai >= 1.40.0 is installed for image config support")
        except (AttributeError, TypeError) as e:
            # ImageConfig doesn't exist or can't be created - skip it
            self._log(f"⚠️ Could not create ImageConfig: {e}")
            self._log("💡 Image generation will proceed without aspect_ratio/image_size control")
            self._log("💡 Ensure google-genai >= 1.40.0 is installed for full image config support")

        config = types.GenerateContentConfig(**config_kwargs)

        self._log("🎛️ Generation parameters:")
        self._log(f"  • Model: {model}")
        self._log(f"  • Aspect ratio: {aspect_ratio}")
        self._log(f"  • Image size: {image_size}")
        self._log(f"  • Temperature: {temperature}")
        self._log(f"  • Top-p: {top_p}")
        self._log(f"  • Google Search: {'Enabled' if use_google_search else 'Disabled'}")
        self._log(f"  • Input images: {len(pil_images)}")

        # Make API call - matching notebook pattern
        self._log("🧠 Calling Gemini 3 Pro generate_content API...")
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
            import traceback

            self._log(traceback.format_exc())
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
                    art = self._create_image_artifact(image_bytes, mime_type)
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
                            art = self._create_image_artifact(image_bytes, mime_type)
                            all_images.append(art)
                    except Exception:
                        pass
                else:
                    self._log(f"ℹ️ Part {idx + 1}: Unknown type (skipping){thought_label}")
            except Exception as e:
                self._log(f"⚠️ Error processing part {idx + 1}: {e}")
                import traceback

                self._log(traceback.format_exc())

        # Set outputs
        if all_images:
            self.parameter_output_values["images"] = all_images
            if len(all_images) == 1:
                self.parameter_output_values["image"] = all_images[0]
                self._log("🖼️ Saved 1 image to both 'image' and 'images' outputs.")
            else:
                self.parameter_output_values["image"] = None
                self._log(f"🖼️ Saved {len(all_images)} image(s) to 'images' output.")
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
    def process(self) -> AsyncResult[None]:
        yield lambda: self._process()

    def _process(self):
        # Clear outputs at the start of each run
        self._reset_outputs()

        if not GOOGLE_INSTALLED:
            self._log(
                "ERROR: Required Google libraries are not installed. Please add 'google-genai' to your library's dependencies."
            )
            return

        if not PIL_INSTALLED:
            self._log("ERROR: Pillow is required to process images. Install 'Pillow' to enable.")
            return

        # Log version information
        self._log(f"📦 google-genai version: {GOOGLE_GENAI_VERSION}")
        if IMAGE_CONFIG_AVAILABLE:
            self._log("✅ ImageConfig is available (aspect_ratio and image_size will be respected)")
        else:
            self._log("⚠️ ImageConfig is NOT available (requires google-genai >= 1.40.0)")
            self._log("   → aspect_ratio and image_size parameters will be ignored")

        # Get input values
        api_provider = self.get_parameter_value("api_provider")
        prompt = self.get_parameter_value("prompt")
        location = self.get_parameter_value("location")
        aspect_ratio = self.get_parameter_value("aspect_ratio")
        image_size = self.get_parameter_value("image_size")
        use_google_search = self.get_parameter_value("use_google_search")
        temperature = self.get_parameter_value("temperature")
        top_p = self.get_parameter_value("top_p")

        reference_images = self.get_parameter_value("reference_images") or []
        auto_image_resize = self.get_parameter_value("auto_image_resize")

        # Backwards compatibility: collect images from deprecated parameters
        object_images = self.get_parameter_value("object_images") or []
        human_images = self.get_parameter_value("human_images") or []

        # Normalize to lists
        if not isinstance(reference_images, list):
            reference_images = [reference_images]
        if not isinstance(object_images, list):
            object_images = [object_images]
        if not isinstance(human_images, list):
            human_images = [human_images]

        # Concatenate all images (deprecated params appended to reference_images)
        all_images = reference_images + object_images + human_images

        # Model name is the same for both APIs
        model = "gemini-3-pro-image-preview"

        self._log(f"📡 Using API provider: {api_provider}")
        self._log(f"🤖 Model: {model}")

        # Validate inputs
        if not prompt and not all_images:
            self._log("❌ Provide at least a prompt or reference images.")
            return

        try:
            # Authenticate based on API provider choice
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
                service_account_file = GriptapeNodes.SecretsManager().get_secret(f"{self.SERVICE_ACCOUNT_FILE_PATH}")
                project_id = None
                credentials = None

                if service_account_file and os.path.exists(service_account_file):
                    self._log("🔑 Using service account file for authentication.")
                    try:
                        # Set environment variable so genai.Client can find credentials
                        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_file
                        project_id = self._get_project_id(service_account_file)
                        credentials = service_account.Credentials.from_service_account_file(
                            service_account_file, scopes=["https://www.googleapis.com/auth/cloud-platform"]
                        )
                        self._log(f"✅ Service account authentication successful for project: {project_id}")
                    except Exception as e:
                        self._log(f"❌ Service account file authentication failed: {e}")
                        raise
                else:
                    # Fall back to individual credentials from settings
                    self._log("🔑 Service account file not found, using individual credentials from settings.")
                    project_id = GriptapeNodes.SecretsManager().get_secret(f"{self.PROJECT_ID}")
                    credentials_json = GriptapeNodes.SecretsManager().get_secret(f"{self.CREDENTIALS_JSON}")

                    if not project_id:
                        raise ValueError(
                            "❌ GOOGLE_CLOUD_PROJECT_ID must be set in library settings when not using a service account file or API key."
                        )

                    if credentials_json:
                        try:
                            cred_dict = json.loads(credentials_json)
                            credentials = service_account.Credentials.from_service_account_info(
                                cred_dict, scopes=["https://www.googleapis.com/auth/cloud-platform"]
                            )
                            # For JSON credentials, write to temp file so genai.Client can use it
                            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                                json.dump(cred_dict, f)
                                temp_cred_file = f.name
                            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_cred_file
                            self._log("✅ JSON credentials authentication successful.")
                        except Exception as e:
                            self._log(f"❌ JSON credentials authentication failed: {e}")
                            raise
                    else:
                        self._log("🔑 Using Application Default Credentials (e.g., gcloud auth).")

                if not project_id:
                    raise ValueError("Could not determine project ID from credentials.")

                self._log(f"Project ID: {project_id}")
                self._log("Initializing Vertex AI...")
                aiplatform.init(project=project_id, location=location, credentials=credentials)

                self._log("Initializing Generative AI Client (Vertex AI)...")
                client = genai.Client(vertexai=True, project=project_id, location=location)

            self._log("🚀 Starting Gemini 3 Pro image generation...")
            self._generate_and_process(
                client=client,
                model=model,
                prompt=prompt,
                input_images=all_images,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                use_google_search=use_google_search,
                temperature=temperature,
                top_p=top_p,
                auto_image_resize=auto_image_resize,
            )

        except ValueError as e:
            self._log(f"❌ CONFIGURATION ERROR: {e}")
            self._log("💡 Please set up credentials in the library settings:")
            self._log("   For AI Studio API: GOOGLE_API_KEY (get from https://aistudio.google.com/apikey)")
            self._log("   For Vertex AI: GOOGLE_SERVICE_ACCOUNT_FILE_PATH (path to service account JSON)")
            self._log("   OR GOOGLE_CLOUD_PROJECT_ID + GOOGLE_APPLICATION_CREDENTIALS_JSON")
        except Exception as e:
            self._log(f"❌ Error: {e}")
            import traceback

            self._log(traceback.format_exc())
            # Ensure stale outputs aren't left behind on errors
            self.parameter_output_values["image"] = None
            self.parameter_output_values["images"] = []
            self.parameter_output_values["text"] = ""
