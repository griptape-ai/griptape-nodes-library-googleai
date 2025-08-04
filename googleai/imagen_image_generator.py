import os
import time
from typing import Any, ClassVar
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterGroup
from griptape_nodes.exe_types.node_types import ControlNode, AsyncResult
from griptape_nodes.traits.options import Options
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

import json
from rich.pretty import pprint

try:
    from google import genai
    from google.oauth2 import service_account
    from google.cloud import aiplatform, storage
    from google.genai import types
    GOOGLE_INSTALLED = True
except ImportError:
    GOOGLE_INSTALLED = False

class VertexAIImageGenerator(ControlNode):
    # Class-level cache for GCS clients
    _gcs_client_cache: ClassVar[dict[str, Any]] = {}
    
    # Service constants for configuration
    SERVICE = "GoogleAI"
    SERVICE_ACCOUNT_FILE_PATH = "GOOGLE_SERVICE_ACCOUNT_FILE_PATH"
    PROJECT_ID = "GOOGLE_CLOUD_PROJECT_ID"
    CREDENTIALS_JSON = "GOOGLE_APPLICATION_CREDENTIALS_JSON"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)    

        self.add_parameter(
            Parameter(
                name="location",
                type="str",
                tooltip="Google Cloud location for the generation job.",
                default_value="us-central1",
                traits=[Options(choices=[
                    "us-central1",
                    "us-east1", 
                    "us-west1",
                    "europe-west1",
                    "europe-west4",
                    "asia-east1"
                ])],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        
        self.add_parameter(
            Parameter(
                name="prompt",
                input_types=["str"],
                type="str",
                output_type="str",
                tooltip="The text prompt for the image.",
                ui_options={"multiline": True, "placeholder_text": "The text prompt for the image."},
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="model",
                type="str",
                tooltip="The Imagen model to use for image generation.",
                default_value="imagen-3.0-generate-002",
                traits=[Options(choices=[
                    "imagen-4.0-generate-preview-06-06",
                    "imagen-4.0-fast-generate-preview-06-06", 
                    "imagen-4.0-ultra-generate-preview-06-06",
                    "imagen-3.0-generate-002",
                    "imagen-3.0-generate-001",
                    "imagen-3.0-fast-generate-001",
                    "imagen-3.0-capability-001",
                    "imagegeneration@006",
                    "imagegeneration@005",
                    "imagegeneration@002"
                ])],
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="number_of_images",
                type="int",
                tooltip="Required. The number of images to generate.",
                default_value=1,
                ui_options={"hide": True},
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="seed",
                type="int",
                tooltip="Optional. The random seed for image generation. This isn't available when addWatermark is set to true.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="negative_prompt",
                type="str",
                tooltip="Optional. A description of what to discourage in the generated images. Not supported by imagen-3.0-generate-002 and newer models.",
                ui_options={"multiline": True},
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="aspect_ratio",
                type="str",
                tooltip="Optional. The aspect ratio for the image.",
                default_value="1:1",
                traits=[Options(choices=["1:1", "16:9", "9:16", "4:3", "3:4"])],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="enhance_prompt",
                type="bool",
                tooltip="Optional. Whether to use the prompt rewriting logic.",
                default_value=True,
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

        with ParameterGroup(name="Advanced") as advanced_group:

            Parameter(
                name="output_mime_type",
                type="str",
                tooltip="Optional. The image format that the output should be saved as.",
                default_value="image/jpeg",
                traits=[Options(choices=["image/png", "image/jpeg"])],
                allowed_modes={ParameterMode.PROPERTY},
            )

            Parameter(
                name="language",
                type="str",
                tooltip="Optional. The language of the text prompt for the image.",
                default_value="auto",
                traits=[Options(choices=["auto", "en", "zh", "zh-CN", "zh-TW", "hi", "ja", "ko", "pt", "es"])],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )

            Parameter(
                name="add_watermark",
                type="bool",
                tooltip="Optional. Add a watermark to the generated image.",
                default_value=False,
                allowed_modes={ParameterMode.PROPERTY},
            )

            Parameter(
                name="safety_filter_level",
                type="str",
                tooltip="Optional. Adds a filter level to safety filtering.",
                default_value="block_medium_and_above",
                traits=[Options(choices=["block_low_and_above", "block_medium_and_above", "block_only_high", "block_none"])],
                allowed_modes={ParameterMode.PROPERTY},
            )

            Parameter(
                name="person_generation",
                type="str",
                tooltip="Optional. Allow generation of people by the model.",
                default_value="allow_adult",
                traits=[Options(choices=["dont_allow", "allow_adult", "allow_all"])],
                allowed_modes={ParameterMode.PROPERTY},
            )

        advanced_group.ui_options = {"collapsed": True}  # Hide the advanced group by default.
        self.add_node_element(advanced_group)



        self.add_parameter(
            Parameter(
                name="image",
                tooltip="Generated image with cached data",
                output_type="ImageUrlArtifact",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        with ParameterGroup(name="Logs") as logs_group:
            Parameter(
                name="include_details",
                type="bool",
                default_value=False,
                tooltip="Include extra details.",
            )

            Parameter(
                name="logs",
                type="str",
                tooltip="Displays processing logs and detailed events if enabled.",
                ui_options={"multiline": True, "placeholder_text": "Logs"},
                allowed_modes={ParameterMode.OUTPUT},
            )

        logs_group.ui_options = {"hide": True}  # Hide the logs group by default.
        self.add_node_element(logs_group)

    def _log(self, message: str):
        """Append a message to the logs output parameter."""
        self.append_value_to_parameter("logs", message + "\n")

    def _get_project_id(self, service_account_file: str) -> str:
        """Read the project_id from the service account JSON file."""
        if not os.path.exists(service_account_file):
            raise FileNotFoundError(f"Service account file not found: {service_account_file}")
        
        with open(service_account_file, 'r') as f:
            service_account_info = json.load(f)
        
        project_id = service_account_info.get('project_id')
        if not project_id:
            raise ValueError("No 'project_id' found in the service account file.")
        
        return project_id

    def _get_gcs_client(self, project_id: str, credentials):
        """Get a cached or new GCS client."""
        if project_id in self._gcs_client_cache:
            return self._gcs_client_cache[project_id]
        else:
            client = storage.Client(project=project_id, credentials=credentials)
            self._gcs_client_cache[project_id] = client
            return client

    def _download_from_gcs(self, gcs_uri: str, project_id: str, credentials) -> bytes:
        """Download video from GCS URI and return bytes."""
        self._log(f"📥 Downloading from GCS URI: {gcs_uri}")
        
        if not gcs_uri.startswith('gs://'):
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")
        
        path_parts = gcs_uri[5:].split('/', 1)
        bucket_name = path_parts[0]
        blob_path = path_parts[1]
        
        storage_client = self._get_gcs_client(project_id, credentials)
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        return blob.download_as_bytes()

    def _create_image_artifact(
        self, image_bytes: bytes, output_format: str
    ) -> ImageUrlArtifact:
        """Create ImageUrlArtifact using StaticFilesManager for efficient storage."""
        try:
            # Generate unique filename with timestamp and hash
            import hashlib

            timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
            content_hash = hashlib.md5(image_bytes).hexdigest()[
                :8
            ]  # Short hash of content
            file_extension = output_format.lower().split('/')[1]
            filename = (
                f"VertexAIImageGenerator_{timestamp}_{content_hash}.{file_extension}"
            )

            # Save to managed file location and get URL
            static_url = GriptapeNodes.StaticFilesManager().save_static_file(
                image_bytes, filename
            )

            return ImageUrlArtifact(value=static_url, name=f"text_to_image_{timestamp}")
        except Exception as e:
            raise ValueError(f"Failed to create image artifact: {str(e)}")

    def _generate_and_process_image(self, client, model, prompt, number_of_images, seed, negative_prompt, 
                                   aspect_ratio, output_mime_type, language, add_watermark, 
                                   safety_filter_level, person_generation, enhance_prompt) -> None:
        """Generate image and process result - called via yield."""
        try:
            image = client.models.generate_images(
                model=model, 
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=number_of_images,
                    seed=seed,
                    negative_prompt=negative_prompt,
                    aspect_ratio=aspect_ratio,
                    output_mime_type=output_mime_type,
                    language=language,
                    add_watermark=add_watermark,
                    safety_filter_level=safety_filter_level,
                    person_generation=person_generation,
                    enhancePrompt=enhance_prompt
                )
            )

            self._log("✅ Image generation completed!")

            # Process the generated images
            if hasattr(image, 'generated_images') and image.generated_images:
                self._log(f"Processing {len(image.generated_images)} generated image(s)...")
                
                # Process the first image
                gen_image = image.generated_images[0]
                if hasattr(gen_image, 'image'):
                    img_obj = gen_image.image
                    
                    # Access image bytes using the working method
                    if hasattr(img_obj, 'image_bytes'):
                        image_bytes = img_obj.image_bytes
                        self._log(f"✅ Retrieved image bytes: {len(image_bytes)} bytes")
                        
                        # Create the image artifact
                        generated_image = self._create_image_artifact(image_bytes, output_mime_type)
                        self._log(f"✅ Created image artifact: {generated_image}")
                        
                        # Set the output parameter
                        self.parameter_output_values["image"] = generated_image
                    else:
                        self._log("❌ Image object does not have image_bytes attribute")
                else:
                    self._log("❌ Generated image does not have 'image' attribute")
            else:
                self._log("❌ No generated images found in response")
        except Exception as e:
            self._log(f"❌ An unexpected error occurred during image generation: {e}")
            import traceback
            self._log(traceback.format_exc())
            raise

    def process(self) -> AsyncResult[None]:
        yield lambda: self._process()

    def _process(self):
        if not GOOGLE_INSTALLED:
            self.append_value_to_parameter("logs", "ERROR: Required libraries are not installed. Please add 'google' to your library's dependencies.")
            return
        
        # Get input values
        prompt = self.get_parameter_value("prompt")
        model = self.get_parameter_value("model")
        number_of_images = self.get_parameter_value("number_of_images")
        seed = self.get_parameter_value("seed")
        negative_prompt = self.get_parameter_value("negative_prompt")
        aspect_ratio = self.get_parameter_value("aspect_ratio")
        output_mime_type = self.get_parameter_value("output_mime_type")
        language = self.get_parameter_value("language")
        add_watermark = self.get_parameter_value("add_watermark")
        location = self.get_parameter_value("location")
        safety_filter_level = self.get_parameter_value("safety_filter_level")
        person_generation = self.get_parameter_value("person_generation")
        enhance_prompt = self.get_parameter_value("enhance_prompt")

        # Validate inputs
        if not prompt:
            self._log("ERROR: Prompt is a required input.")
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
                    raise ValueError("❌ GOOGLE_CLOUD_PROJECT_ID must be set in library settings when not using a service account file.")
                
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

            self._log("Starting image generation...\n")

            # Call the image generation method directly
            self._generate_and_process_image(
                client, model, prompt, number_of_images, seed, negative_prompt, 
                aspect_ratio, output_mime_type, language, add_watermark, 
                safety_filter_level, person_generation, enhance_prompt
            )

        except ValueError as e:
            self._log(f"❌ CONFIGURATION ERROR: {e}")
            self._log("💡 Please set up Google Cloud credentials in the library settings:")
            self._log("   - GOOGLE_SERVICE_ACCOUNT_FILE_PATH (path to service account JSON)")
            self._log("   - OR GOOGLE_CLOUD_PROJECT_ID + GOOGLE_APPLICATION_CREDENTIALS_JSON")
        except Exception as e:
            self._log(f"❌ An unexpected error occurred: {e}")
            import traceback
            self._log(traceback.format_exc())



