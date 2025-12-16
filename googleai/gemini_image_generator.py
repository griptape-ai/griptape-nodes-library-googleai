import logging
import time
from typing import Any

from griptape.artifacts import (
    BlobArtifact,
    ImageArtifact,
    ImageUrlArtifact,
    TextArtifact,
)

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

try:
    import requests

    REQUESTS_INSTALLED = True
except Exception:
    REQUESTS_INSTALLED = False

try:
    from google import genai
    from google.cloud import aiplatform
    from google.genai import types

    GOOGLE_INSTALLED = True
except ImportError:
    GOOGLE_INSTALLED = False

from googleai_utils import GoogleAuthHelper, validate_and_maybe_shrink_image

logger = logging.getLogger("griptape_nodes_library_googleai")

MODELS = []


class GeminiImageGenerator(ControlNode):
    """Gemini-only image generation node for Vertex AI (Gemini 2.5 Flash Image).

    - Location is 'global'.
    - Supports text prompt + up to 3 input images (≤ 7 MB each; png/jpeg/webp)
      and up to 3 input documents (≤ 50 MB each; pdf/txt).
    - Uses GenerateContent with response_modalities=["IMAGE","TEXT"].
    - Returns the FIRST generated image as ImageUrlArtifact (parameter 'image').
    - Streams/logs info; captures any returned text to logs for visibility.
    """

    SERVICE = "GoogleAI"

    # Model constraints: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash-image
    MAX_PROMPT_IMAGES = 3
    MAX_PROMPT_DOCS = 3
    MAX_IMAGE_BYTES = 7 * 1024 * 1024  # 7 MB
    MAX_DOC_BYTES = 7 * 1024 * 1024  # 7 MB (direct upload, not Cloud Storage)
    ALLOWED_IMAGE_MIME = {"image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"}
    ALLOWED_DOC_MIME = {"application/pdf", "text/plain"}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # ===== Core configuration =====
        self.add_parameter(
            Parameter(
                name="prompt",
                input_types=["str"],
                type="str",
                output_type="str",
                tooltip="User prompt for generation.",
                ui_options={"multiline": True, "placeholder_text": "Enter prompt..."},
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="location",
                type="str",
                tooltip="Google Cloud location for Gemini image generation.",
                default_value="us-central1",
                traits=[Options(choices=["us-central1", "europe-west1", "asia-southeast1", "global"])],
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

        # ===== Inputs: images & documents =====
        self.add_parameter(
            ParameterList(
                name="input_images",
                tooltip="Up to 3 input images (png/jpeg/webp, ≤ 7 MB each). These visual references are used by the model to guide image generation, similar to image-to-image generation.",
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
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            ParameterList(
                name="input_files",
                tooltip="Up to 3 input files (pdf/txt, ≤ 50 MB each). Text content from these documents is extracted and included as additional context in the prompt to guide image generation.",
                input_types=["BlobArtifact", "TextArtifact"],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # Image configuration
        self.add_parameter(
            Parameter(
                name="aspect_ratio",
                type="str",
                tooltip="Aspect ratio for generated images.",
                default_value="16:9",
                traits=[Options(choices=["1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"])],
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

        # Sampling / candidates
        self.add_parameter(
            ParameterFloat(
                name="temperature",
                tooltip="Sampling temperature for image generation (0.0–2.0). Higher values increase randomness.",
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
        self.add_parameter(
            Parameter(
                name="candidate_count",
                type="int",
                tooltip="Number of candidates to sample (1–8).",
                default_value=1,
                allowed_modes={ParameterMode.PROPERTY},
                hide=True,
            )
        )

        # ===== Output =====
        self.add_parameter(
            Parameter(
                name="image",
                tooltip="Generated image with cached data",
                output_type="ImageUrlArtifact",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="images",
                tooltip="All generated images (as URL).",
                output_type="list[ImageUrlArtifact]",
                allowed_modes={ParameterMode.OUTPUT},
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

    # ---------- Utilities ----------
    def _log(self, message: str):
        logger.info(message)
        self.append_value_to_parameter("logs", message + "\n")

    def _reset_outputs(self) -> None:
        """Clear output parameters so stale values don't persist across re-adds/reruns."""
        try:
            self.parameter_output_values["image"] = None
            self.parameter_output_values["images"] = []
            self.parameter_output_values["logs"] = ""
        except Exception:
            # Be defensive if the base class changes how outputs are stored
            pass


    def _create_image_artifact(self, image_bytes: bytes, mime_type: str) -> ImageUrlArtifact:
        import hashlib

        timestamp = int(time.time() * 1000)
        content_hash = hashlib.md5(image_bytes).hexdigest()[:8]
        ext = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/webp": "webp",
        }.get(mime_type, "png")
        filename = f"GeminiImage_{timestamp}_{content_hash}.{ext}"
        static_url = GriptapeNodes.StaticFilesManager().save_static_file(image_bytes, filename)
        return ImageUrlArtifact(value=static_url, name=f"gemini_image_{timestamp}")

    # ---- Artifact → (bytes, mime) helpers ----
    def _fetch_image_url_bytes(self, url: str) -> tuple[bytes, str]:
        if not REQUESTS_INSTALLED:
            raise RuntimeError("`requests` is required to fetch ImageUrlArtifact URLs.")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        mime = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        return resp.content, mime

    def _image_artifact_to_bytes_mime(self, art: Any) -> tuple[bytes, str]:
        if isinstance(art, ImageArtifact):
            # ImageArtifact.value is expected to be raw bytes; may have mime_type attr
            mime = getattr(art, "mime_type", None) or "image/png"
            return art.value, mime
        if isinstance(art, ImageUrlArtifact):
            return self._fetch_image_url_bytes(art.value)
        raise TypeError("Unsupported image artifact type.")

    def _file_artifact_to_bytes_mime(self, art: Any) -> tuple[bytes, str]:
        if isinstance(art, TextArtifact):
            data = (art.value or "").encode("utf-8")
            return data, "text/plain"
        if isinstance(art, BlobArtifact):
            data = art.value
            mime = getattr(art, "mime_type", None)
            if not mime:
                # Fallback guess by simple sniff
                mime = "application/pdf" if getattr(art, "name", "").lower().endswith(".pdf") else "text/plain"
            return data, mime
        raise TypeError("Unsupported file artifact type.")


    # ---------- Core generation ----------
    def _generate_and_process(
        self,
        client,
        model,
        prompt,
        input_images,
        input_files,
        temperature,
        top_p,
        candidate_count,
        aspect_ratio,
        auto_image_resize,
    ):
        # Build contents list for SDK
        contents: list = []

        if prompt:
            contents.append(prompt)

        # Images (max 3, ≤ 7 MB each, allowed mimes)
        images = input_images or []
        if not isinstance(images, list):
            images = [images]
        kept = 0
        for img_idx, img_art in enumerate(images):
            if kept >= self.MAX_PROMPT_IMAGES:
                self._log("ℹ️ Only the first 3 input images are used.")
                break
            try:
                b, mime = self._image_artifact_to_bytes_mime(img_art)
                img_name = getattr(img_art, "name", f"image_{img_idx + 1}")
                b, mime = validate_and_maybe_shrink_image(
                    image_bytes=b,
                    mime_type=mime,
                    image_name=img_name,
                    allowed_mimes=self.ALLOWED_IMAGE_MIME,
                    byte_limit=self.MAX_IMAGE_BYTES,
                    auto_image_resize=auto_image_resize,
                    log_func=self._log,
                )
                # SDK format: types.Part with inline_data
                contents.append(types.Part.from_bytes(data=b, mime_type=mime))
                kept += 1
            except Exception as e:
                self._log(f"⚠️ Skipping image due to error: {e}")

        # Documents (max 3, ≤ 50 MB each, allowed mimes)
        docs = input_files or []
        if not isinstance(docs, list):
            docs = [docs]
        kept = 0
        for doc_idx, doc_art in enumerate(docs):
            if kept >= self.MAX_PROMPT_DOCS:
                self._log("ℹ️ Only the first 3 input files are used.")
                break
            try:
                b, mime = self._file_artifact_to_bytes_mime(doc_art)
                if mime not in self.ALLOWED_DOC_MIME:
                    doc_name = getattr(doc_art, "name", f"document_{doc_idx + 1}")
                    error_msg = f"❌ Document '{doc_name}' has unsupported MIME type: {mime}. Supported types: {', '.join(self.ALLOWED_DOC_MIME)}"
                    self._log(error_msg)
                    raise ValueError(error_msg)
                if len(b) > self.MAX_DOC_BYTES:
                    doc_name = getattr(doc_art, "name", f"document_{doc_idx + 1}")
                    size_mb = len(b) / (1024 * 1024)
                    error_msg = f"❌ Document '{doc_name}' size {size_mb:.1f} MB exceeds maximum allowed size of 50 MB"
                    self._log(error_msg)
                    raise ValueError(error_msg)
                # SDK format: types.Part with inline_data
                contents.append(types.Part.from_bytes(data=b, mime_type=mime))
                kept += 1
            except Exception as e:
                self._log(f"⚠️ Skipping file due to error: {e}")

        # Validate candidate count
        original_candidates = int(candidate_count or 1)
        eff_candidates = max(1, min(original_candidates, 8))

        if original_candidates != eff_candidates:
            self._log(f"⚠️ Candidate count adjusted from {original_candidates} to {eff_candidates} (valid range: 1-8)")

        self._log("🎛️ Generation parameters:")
        self._log(f"  • Temperature: {temperature}")
        self._log(f"  • Top-p: {top_p}")
        self._log(f"  • Candidate count: {eff_candidates}")
        self._log(f"  • Aspect ratio: {aspect_ratio}")

        # Build generation config
        config = types.GenerateContentConfig(
            temperature=float(temperature),
            top_p=float(top_p),
            candidate_count=eff_candidates,
            response_modalities=["TEXT", "IMAGE"],
        )

        self._log("🧠 Calling Gemini generateContent API...")

        # Call the SDK
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        self._log("✅ Generation complete.")

        # Parse outputs from SDK response
        all_images = []
        if response.candidates:
            for cand in response.candidates:
                if cand.content and cand.content.parts:
                    for part in cand.content.parts:
                        # Text logs
                        if part.text:
                            self._log(part.text)

                        # Inline images - SDK returns inline_data as an object
                        if hasattr(part, "inline_data") and part.inline_data:
                            inline_data = part.inline_data
                            mime = getattr(inline_data, "mime_type", "image/png")
                            data = getattr(inline_data, "data", None)
                            if mime.startswith("image/") and data:
                                art = self._create_image_artifact(data, mime)
                                all_images.append(art)

        # Save all images to outputs
        if all_images:
            self.parameter_output_values["images"] = all_images

            # If there's exactly one image, also set it in the single image parameter
            if len(all_images) == 1:
                self.parameter_output_values["image"] = all_images[0]
                self._log("🖼️ Received 1 image. Saved to both 'image' and 'images' outputs.")
            else:
                # Multiple images: clear the single-image output to avoid stale values
                self.parameter_output_values["image"] = None
                self._log(f"🖼️ Received {len(all_images)} image(s). Saved to the 'images' output.")
        else:
            # No images returned: clear outputs
            self.parameter_output_values["image"] = None
            self.parameter_output_values["images"] = []
            self._log("ℹ️ No image outputs returned.")

    # ---------- Node entrypoints ----------
    def process(self) -> AsyncResult[None]:
        yield lambda: self._process()

    def _process(self):
        # Clear outputs at the start of each run
        self._reset_outputs()
        if not GOOGLE_INSTALLED:
            self._log("ERROR: Missing Google libraries. Install `google-genai`, `google-cloud-aiplatform`.")
            return

        # Inputs
        prompt = self.get_parameter_value("prompt")
        location = self.get_parameter_value("location")

        # Model is hardcoded (preview version is deprecated)
        model = "gemini-2.5-flash-image"
        aspect_ratio = self.get_parameter_value("aspect_ratio")

        input_images = self.get_parameter_value("input_images")
        input_files = self.get_parameter_value("input_files")
        auto_image_resize = self.get_parameter_value("auto_image_resize")

        temperature = self.get_parameter_value("temperature")
        top_p = self.get_parameter_value("top_p")
        candidate_count = self.get_parameter_value("candidate_count")

        if not prompt and not input_images and not input_files:
            # Clear outputs on validation failure
            self.parameter_output_values["image"] = None
            self.parameter_output_values["images"] = []
            self._log("❌ Provide at least a prompt, an image, or a file.")
            return

        try:
            # Use GoogleAuthHelper for authentication
            credentials, project_id = GoogleAuthHelper.get_credentials_and_project(
                GriptapeNodes.SecretsManager(),
                log_func=self._log
            )

            self._log(f"Project ID: {project_id}")
            self._log("Initializing Vertex AI...")
            aiplatform.init(project=project_id, location=location, credentials=credentials)

            self._log("Initializing Generative AI Client (Vertex AI)...")
            client = genai.Client(vertexai=True, project=project_id, location=location)

            self._log("🚀 Starting Gemini image generation...")
            self._generate_and_process(
                client=client,
                model=model,
                prompt=prompt,
                input_images=input_images,
                input_files=input_files,
                temperature=temperature,
                top_p=top_p,
                candidate_count=candidate_count,
                aspect_ratio=aspect_ratio,
                auto_image_resize=auto_image_resize,
            )

        except ValueError as e:
            self._log(f"❌ CONFIGURATION ERROR: {e}")
            self._log("💡 Please set up Google Cloud credentials in the library settings:")
            self._log("   - GOOGLE_WORKLOAD_IDENTITY_CONFIG_PATH (recommended, path to workload identity config)")
            self._log("   - OR GOOGLE_SERVICE_ACCOUNT_FILE_PATH (path to service account JSON)")
            self._log("   - OR GOOGLE_CLOUD_PROJECT_ID + GOOGLE_APPLICATION_CREDENTIALS_JSON")
        except Exception as e:
            self._log(f"❌ Error: {e}")
            import traceback

            self._log(traceback.format_exc())
            # Ensure stale outputs aren't left behind on errors
            self.parameter_output_values["image"] = None
            self.parameter_output_values["images"] = []
