import os
import time
import json
from typing import Any, ClassVar, List

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterGroup, ParameterList
from griptape_nodes.exe_types.node_types import ControlNode, AsyncResult
from griptape_nodes.traits.options import Options
from griptape.artifacts import (
    ImageArtifact,
    ImageUrlArtifact,
    BlobArtifact,
    TextArtifact,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

try:
    from google import genai
    from google.oauth2 import service_account
    from google.cloud import aiplatform
    from google.genai import types
    GOOGLE_INSTALLED = True
except ImportError:
    GOOGLE_INSTALLED = False

# Optional dependency for fetching ImageUrlArtifact bytes
try:
    import requests
    REQUESTS_INSTALLED = True
except Exception:
    REQUESTS_INSTALLED = False


class GeminiImageGenerator(ControlNode):
    """
    Gemini-only image generation node for Vertex AI (Gemini 2.5 Flash Image Preview).

    - Location is 'global'.
    - Supports text prompt + up to 3 input images (≤ 7 MB each; png/jpeg/webp)
      and up to 3 input documents (≤ 50 MB each; pdf/txt).
    - Uses GenerateContent with response_modalities=["IMAGE","TEXT"].
    - Returns the FIRST generated image as ImageUrlArtifact (parameter 'image').
    - Streams/logs info; captures any returned text to logs for visibility.
    """

    SERVICE = "GoogleAI"
    SERVICE_ACCOUNT_FILE_PATH = "GOOGLE_SERVICE_ACCOUNT_FILE_PATH"
    PROJECT_ID = "GOOGLE_CLOUD_PROJECT_ID"
    CREDENTIALS_JSON = "GOOGLE_APPLICATION_CREDENTIALS_JSON"

    # Model constraints (from model card)
    MAX_PROMPT_IMAGES = 3
    MAX_PROMPT_DOCS = 3
    MAX_IMAGE_BYTES = 7 * 1024 * 1024        # 7 MB
    MAX_DOC_BYTES = 50 * 1024 * 1024         # 50 MB
    ALLOWED_IMAGE_MIME = {"image/png", "image/jpeg", "image/webp"}
    ALLOWED_DOC_MIME = {"application/pdf", "text/plain"}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # ===== Core configuration =====
        self.add_parameter(
            Parameter(
                name="location",
                type="str",
                tooltip="Google Cloud location (Gemini uses 'global').",
                default_value="global",
                traits=[Options(choices=["global"])],
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

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
                name="model",
                type="str",
                tooltip="Gemini model (fixed for this node).",
                default_value="gemini-2.5-flash-image-preview",
                traits=[Options(choices=["gemini-2.5-flash-image-preview"])],
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

        # Sampling / candidates
        self.add_parameter(
            Parameter(
                name="temperature",
                type="float",
                tooltip="Sampling temperature (0.0–2.0).",
                default_value=1.0,
                allowed_modes={ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="top_p",
                type="float",
                tooltip="Top-p nucleus sampling (0.0–1.0).",
                default_value=0.95,
                allowed_modes={ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="candidate_count",
                type="int",
                tooltip="Number of candidates to sample (1–8).",
                default_value=1,
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

        # ===== Inputs: images & documents =====
        self.add_parameter(
            ParameterList(
                name="input_images",
                tooltip="Up to 3 input images (png/jpeg/webp, ≤ 7 MB each).",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            ParameterList(
                name="input_files",
                tooltip="Up to 3 input files (pdf/txt, ≤ 50 MB each).",
                input_types=["BlobArtifact", "TextArtifact"],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # ===== Output =====
        self.add_parameter(
            Parameter(
                name="image",
                tooltip="First generated image (as URL).",
                output_type="ImageUrlArtifact",
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

    # ---------- Utilities ----------
    def _log(self, message: str):
        self.append_value_to_parameter("logs", message + "\n")

    def _get_project_id(self, service_account_file: str) -> str:
        if not os.path.exists(service_account_file):
            raise FileNotFoundError(f"Service account file not found: {service_account_file}")
        with open(service_account_file, "r") as f:
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
    self, client, model, prompt, input_images, input_files, temperature, top_p, candidate_count
):
        # Build parts respecting model limits
        parts: List[types.Part] = []

        if prompt:
            parts.append(types.Part.from_text(text=prompt))

        # Images (max 3, ≤ 7 MB each, allowed mimes)
        images = input_images or []
        if not isinstance(images, list):
            images = [images]
        kept = 0
        for img_art in images:
            if kept >= self.MAX_PROMPT_IMAGES:
                self._log("ℹ️ Only the first 3 input images are used.")
                break
            try:
                b, mime = self._image_artifact_to_bytes_mime(img_art)
                if mime not in self.ALLOWED_IMAGE_MIME:
                    self._log(f"⚠️ Skipping image with unsupported MIME: {mime}")
                    continue
                if len(b) > self.MAX_IMAGE_BYTES:
                    self._log(f"⚠️ Skipping image over 7 MB (size={len(b)} bytes).")
                    continue
                parts.append(types.Part.from_bytes(data=b, mime_type=mime))
                kept += 1
            except Exception as e:
                self._log(f"⚠️ Skipping image due to error: {e}")

        # Documents (max 3, ≤ 50 MB each, allowed mimes)
        docs = input_files or []
        if not isinstance(docs, list):
            docs = [docs]
        kept = 0
        for doc_art in docs:
            if kept >= self.MAX_PROMPT_DOCS:
                self._log("ℹ️ Only the first 3 input files are used.")
                break
            try:
                b, mime = self._file_artifact_to_bytes_mime(doc_art)
                if mime not in self.ALLOWED_DOC_MIME:
                    self._log(f"⚠️ Skipping file with unsupported MIME: {mime}")
                    continue
                if len(b) > self.MAX_DOC_BYTES:
                    self._log(f"⚠️ Skipping file over 50 MB (size={len(b)} bytes).")
                    continue
                parts.append(types.Part.from_bytes(data=b, mime_type=mime))
                kept += 1
            except Exception as e:
                self._log(f"⚠️ Skipping file due to error: {e}")

        contents = [types.Content(role="user", parts=parts)]
        eff_candidates = max(1, min(int(candidate_count or 1), 8))

        config = types.GenerateContentConfig(
            temperature=float(temperature or 1.0),
            top_p=float(top_p or 0.95),
            candidate_count=eff_candidates,        # 1–8
            response_modalities=["IMAGE", "TEXT"], # allow both
            # You can add safety_settings here if you expose them as parameters.
        )

        self._log("🧠 Calling Gemini generate_content...")
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        self._log("✅ Generation complete.")

        # Parse outputs
        total_images = 0
        first_image_set = False
        if getattr(response, "candidates", None):
            for ci, cand in enumerate(response.candidates):
                if not getattr(cand, "content", None):
                    continue
                for pi, part in enumerate(cand.content.parts or []):
                    # Text
                    if getattr(part, "text", None):
                        self._log(part.text)
                    # Inline images
                    if getattr(part, "inline_data", None):
                        mime = getattr(part.inline_data, "mime_type", "image/png")
                        data = getattr(part.inline_data, "data", b"")
                        if mime.startswith("image/") and data:
                            total_images += 1
                            if not first_image_set:
                                art = self._create_image_artifact(data, mime)
                                self.parameter_output_values["image"] = art
                                first_image_set = True

        if total_images == 0:
            self._log("ℹ️ No image outputs returned.")
        else:
            self._log(f"🖼️ Received {total_images} image(s). Saved the first to the 'image' output.")


    # ---------- Node entrypoints ----------
    def process(self) -> AsyncResult[None]:
        yield lambda: self._process()

    def _process(self):
        if not GOOGLE_INSTALLED:
            self._log("ERROR: Missing Google AI libraries. Install `google-genai` and `google-cloud-aiplatform`.")
            return

        # Inputs
        prompt = self.get_parameter_value("prompt")
        model = self.get_parameter_value("model")
        location = self.get_parameter_value("location")

        input_images = self.get_parameter_value("input_images")
        input_files = self.get_parameter_value("input_files")

        temperature = self.get_parameter_value("temperature")
        top_p = self.get_parameter_value("top_p")
        candidate_count = self.get_parameter_value("candidate_count")

        if not prompt and not input_images and not input_files:
            self._log("❌ Provide at least a prompt, an image, or a file.")
            return

        try:
            # Auth
            service_account_file = self.get_config_value(service=self.SERVICE, value=self.SERVICE_ACCOUNT_FILE_PATH)
            project_id = None
            credentials = None
            self._log("service account file " + service_account_file)

            if service_account_file and os.path.exists(service_account_file):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_file
                project_id = self._get_project_id(service_account_file)
                credentials = service_account.Credentials.from_service_account_file(service_account_file)
                self._log(f"🔑 Authenticated with service account for project '{project_id}'.")
            else:
                project_id = self.get_config_value(service=self.SERVICE, value=self.PROJECT_ID)
                credentials_json = self.get_config_value(service=self.SERVICE, value=self.CREDENTIALS_JSON)
                if credentials_json:
                    cred_dict = json.loads(credentials_json)
                    credentials = service_account.Credentials.from_service_account_info(cred_dict)
                    self._log(f"🔑 Authenticated with JSON credentials for project '{project_id}'.")
                else:
                    self._log("⚠️ Using Application Default Credentials (e.g., gcloud auth). Ensure ADC has Vertex access.")

            # Init Vertex + client
            aiplatform.init(project=project_id, location=location, credentials=credentials)
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
            )

        except Exception as e:
            self._log(f"❌ Error: {e}")
            import traceback
            self._log(traceback.format_exc())
