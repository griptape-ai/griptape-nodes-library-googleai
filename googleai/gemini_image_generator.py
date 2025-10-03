import os
import time
import json
import base64
from typing import Any, List

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
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request
    GOOGLE_AUTH_INSTALLED = True
except ImportError:
    GOOGLE_AUTH_INSTALLED = False

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
                tooltip="Google Cloud location for Gemini image generation.",
                default_value="us-central1",
                traits=[Options(choices=["us-central1", "europe-west1", "asia-southeast1", "global"])],
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
                tooltip="Gemini model for image generation.",
                default_value="gemini-2.5-flash-image",
                traits=[Options(choices=["gemini-2.5-flash-image", "gemini-2.5-flash-image-preview"])],
                allowed_modes={ParameterMode.PROPERTY},
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
            Parameter(
                name="temperature",
                type="float",
                tooltip="Sampling temperature for image generation (0.0–1.0). Higher values increase randomness.",
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
                ui_options={"hide": True},
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

    def _get_access_token(self, credentials) -> str:
        """Get access token from credentials."""
        if not credentials.valid:
            credentials.refresh(Request())
        return credentials.token

    # ---------- Core generation ----------
    def _generate_and_process(
    self, credentials, project_id, location, model, prompt, input_images, input_files, 
    temperature, top_p, candidate_count, aspect_ratio
):
        # Build parts list for REST API
        parts: List[dict] = []

        if prompt:
            parts.append({"text": prompt})

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
                if mime not in self.ALLOWED_IMAGE_MIME:
                    img_name = getattr(img_art, 'name', f'image_{img_idx + 1}')
                    error_msg = f"❌ Image '{img_name}' has unsupported MIME type: {mime}. Supported types: {', '.join(self.ALLOWED_IMAGE_MIME)}"
                    self._log(error_msg)
                    raise ValueError(error_msg)
                if len(b) > self.MAX_IMAGE_BYTES:
                    img_name = getattr(img_art, 'name', f'image_{img_idx + 1}')
                    size_mb = len(b) / (1024 * 1024)
                    error_msg = f"❌ Image '{img_name}' size {size_mb:.1f} MB exceeds maximum allowed size of 7 MB"
                    self._log(error_msg)
                    raise ValueError(error_msg)
                # REST API format: inline_data with base64
                parts.append({
                    "inline_data": {
                        "mime_type": mime,
                        "data": base64.b64encode(b).decode('utf-8')
                    }
                })
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
                    doc_name = getattr(doc_art, 'name', f'document_{doc_idx + 1}')
                    error_msg = f"❌ Document '{doc_name}' has unsupported MIME type: {mime}. Supported types: {', '.join(self.ALLOWED_DOC_MIME)}"
                    self._log(error_msg)
                    raise ValueError(error_msg)
                if len(b) > self.MAX_DOC_BYTES:
                    doc_name = getattr(doc_art, 'name', f'document_{doc_idx + 1}')
                    size_mb = len(b) / (1024 * 1024)
                    error_msg = f"❌ Document '{doc_name}' size {size_mb:.1f} MB exceeds maximum allowed size of 50 MB"
                    self._log(error_msg)
                    raise ValueError(error_msg)
                # REST API format: inline_data with base64
                parts.append({
                    "inline_data": {
                        "mime_type": mime,
                        "data": base64.b64encode(b).decode('utf-8')
                    }
                })
                kept += 1
            except Exception as e:
                self._log(f"⚠️ Skipping file due to error: {e}")

        # Build REST API request
        original_candidates = int(candidate_count or 1)
        eff_candidates = max(1, min(original_candidates, 8))
        
        if original_candidates != eff_candidates:
            self._log(f"⚠️ Candidate count adjusted from {original_candidates} to {eff_candidates} (valid range: 1-8)")
        
        # Validate parameters for image generation model
        eff_temperature = float(temperature or 1.0)
        if not (0.0 <= eff_temperature <= 1.0):
            error_msg = (
                f"❌ Temperature must be between 0.0 and 1.0 for image generation models. "
                f"Got: {eff_temperature}. The gemini-2.5-flash-image-preview model has "
                f"more restrictive temperature constraints than text generation models."
            )
            self._log(error_msg)
            raise ValueError(error_msg)
        
        eff_top_p = float(top_p or 0.95)
        if not (0.0 <= eff_top_p <= 1.0):
            error_msg = f"❌ Top-p must be between 0.0 and 1.0. Got: {eff_top_p}"
            self._log(error_msg)
            raise ValueError(error_msg)

        # Build REST API payload
        payload = {
            "contents": {
                "role": "USER",
                "parts": parts
            },
            "generation_config": {
                "temperature": eff_temperature,
                "topP": eff_top_p,
                "candidateCount": eff_candidates,
                "response_modalities": ["TEXT", "IMAGE"],
                "image_config": {
                    "aspect_ratio": aspect_ratio
                }
            }
        }

        self._log("🎛️ Generation parameters:")
        self._log(f"  • Temperature: {eff_temperature}")
        self._log(f"  • Top-p: {eff_top_p}")
        self._log(f"  • Candidate count: {eff_candidates}")
        self._log(f"  • Aspect ratio: {aspect_ratio}")
        
        # Make REST API call
        access_token = self._get_access_token(credentials)
        api_endpoint = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model}:generateContent"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        self._log("🧠 Calling Gemini generateContent API...")
        if not REQUESTS_INSTALLED:
            raise RuntimeError("`requests` library is required for REST API calls.")
        
        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        response_data = response.json()
        self._log("✅ Generation complete.")

        # Parse outputs from REST API JSON response
        all_images = []
        candidates = response_data.get("candidates", [])
        for cand in candidates:
            content = cand.get("content", {})
            parts_list = content.get("parts", [])
            for part in parts_list:
                # Text logs
                if "text" in part:
                    self._log(part["text"])

                # Inline images
                if "inlineData" in part or "inline_data" in part:
                    inline_data = part.get("inlineData") or part.get("inline_data", {})
                    mime = inline_data.get("mimeType") or inline_data.get("mime_type", "image/png")
                    data_b64 = inline_data.get("data", "")
                    if mime.startswith("image/") and data_b64:
                        # Decode base64
                        data = base64.b64decode(data_b64)
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
        if not GOOGLE_AUTH_INSTALLED:
            self._log("ERROR: Missing Google auth libraries. Install `google-auth`.")
            return

        # Inputs
        prompt = self.get_parameter_value("prompt")
        model = self.get_parameter_value("model")
        location = self.get_parameter_value("location")
        aspect_ratio = self.get_parameter_value("aspect_ratio")

        input_images = self.get_parameter_value("input_images")
        input_files = self.get_parameter_value("input_files")

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
            # Auth
            service_account_file = self.get_config_value(service=self.SERVICE, value=self.SERVICE_ACCOUNT_FILE_PATH)
            project_id = None
            credentials = None

            if service_account_file and os.path.exists(service_account_file):
                project_id = self._get_project_id(service_account_file)
                credentials = service_account.Credentials.from_service_account_file(
                    service_account_file,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                self._log(f"🔑 Authenticated with service account for project '{project_id}'.")
            else:
                project_id = self.get_config_value(service=self.SERVICE, value=self.PROJECT_ID)
                credentials_json = self.get_config_value(service=self.SERVICE, value=self.CREDENTIALS_JSON)
                if credentials_json:
                    cred_dict = json.loads(credentials_json)
                    credentials = service_account.Credentials.from_service_account_info(
                        cred_dict,
                        scopes=['https://www.googleapis.com/auth/cloud-platform']
                    )
                    project_id = cred_dict.get("project_id")
                    self._log(f"🔑 Authenticated with JSON credentials for project '{project_id}'.")
                else:
                    raise ValueError("No credentials provided. Configure service account file or credentials JSON.")

            if not project_id:
                raise ValueError("Could not determine project ID from credentials.")

            self._log("🚀 Starting Gemini image generation...")
            self._generate_and_process(
                credentials=credentials,
                project_id=project_id,
                location=location,
                model=model,
                prompt=prompt,
                input_images=input_images,
                input_files=input_files,
                temperature=temperature,
                top_p=top_p,
                candidate_count=candidate_count,
                aspect_ratio=aspect_ratio,
            )

        except Exception as e:
            self._log(f"❌ Error: {e}")
            import traceback
            self._log(traceback.format_exc())
            # Ensure stale outputs aren't left behind on errors
            self.parameter_output_values["image"] = None
            self.parameter_output_values["images"] = []
