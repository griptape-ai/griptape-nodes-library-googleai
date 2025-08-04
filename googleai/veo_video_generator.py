import os
import time
import json
import asyncio
from typing import Any, ClassVar
from griptape.artifacts import UrlArtifact, ListArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterGroup
from griptape_nodes.exe_types.node_types import DataNode, AsyncResult
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

# Attempt to import Google libraries
try:
    from google.oauth2 import service_account
    from google.cloud import aiplatform, storage
    from google import genai
    from google.genai.types import GenerateVideosConfig
    GOOGLE_INSTALLED = True
except ImportError:
    GOOGLE_INSTALLED = False


class VideoUrlArtifact(UrlArtifact):
    """
    Artifact that contains a URL to a video.
    """
    def __init__(self, value: str, name: str | None = None):
        super().__init__(value=value, name=name or self.__class__.__name__)

class VeoVideoGenerator(DataNode):
    # Class-level cache for GCS clients
    _gcs_client_cache: ClassVar[dict[str, Any]] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category = "Google AI"
        self.description = "Generates videos using Google's Veo model."

        # Main Parameters
        self.add_parameter(
            Parameter(
                name="prompt",
                type="str",
                tooltip="The text prompt for video generation.",
                ui_options={"multiline": True, "placeholder_text": "Describe the video you want to generate"},
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="model",
                type="str",
                tooltip="The Veo model to use for generation.",
                default_value="veo-3.0-generate-preview",
                traits=[Options(choices=["veo-3.0-generate-preview"])],
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="number_of_videos",
                type="int",
                tooltip="Number of videos to generate.",
                default_value=1,
                traits=[Options(choices=[1, 2])],
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="aspect_ratio",
                type="str",
                tooltip="Aspect ratio of the generated video.",
                default_value="16:9",
                traits=[Options(choices=["16:9", "9:16", "1:1"])],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # Google Config Group
        with ParameterGroup(name="GoogleConfig") as google_config_group:
            Parameter(
                name="service_account_file",
                type="str",
                tooltip="Optional: Path to a Google Cloud service account JSON file. If empty, Application Default Credentials will be used.",
                ui_options={"clickable_file_browser": True},
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )

            Parameter(
                name="project_id",
                type="str",
                tooltip="Google Cloud Project ID. Required if not using a service account file.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )

            Parameter(
                name="location",
                type="str",
                tooltip="Google Cloud location for the generation job.",
                default_value="us-central1",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )

        google_config_group.ui_options = {"collapsed": True}
        self.add_node_element(google_config_group)

        # Output Parameters
        self.add_parameter(
            Parameter(
                name="video_artifacts",
                type="list[VideoUrlArtifact]",
                tooltip="List of generated video artifacts.",
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

        logs_group.ui_options = {"hide": True}
        self.add_node_element(logs_group)

    def _log(self, message: str):
        """Append a message to the logs output parameter."""
        print(message)
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
        
    async def _download_from_gcs_async(self, gcs_uri: str, project_id: str, credentials) -> bytes:
        """Asynchronously download video from GCS URI and return bytes."""
        self._log(f"📥 Downloading from GCS URI: {gcs_uri}")
        
        if not gcs_uri.startswith('gs://'):
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")
        
        path_parts = gcs_uri[5:].split('/', 1)
        bucket_name = path_parts[0]
        blob_path = path_parts[1]
        
        storage_client = self._get_gcs_client(project_id, credentials)
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        # Use asyncio to run the blocking operation in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, blob.download_as_bytes)

    def _process(self) -> None:
        if not GOOGLE_INSTALLED:
            self._log("ERROR: Required Google libraries are not installed. Please add 'google-cloud-aiplatform', 'google-generativeai', 'google-cloud-storage' to your library's dependencies.")
            return

        # Get input values
        service_account_file = self.get_parameter_value("service_account_file")
        user_project_id = self.get_parameter_value("project_id")
        prompt = self.get_parameter_value("prompt")
        model = self.get_parameter_value("model")
        num_videos = self.get_parameter_value("number_of_videos")
        aspect_ratio = self.get_parameter_value("aspect_ratio")
        location = self.get_parameter_value("location")

        # Validate inputs
        if not prompt:
            self._log("ERROR: Prompt is a required input.")
            return

        try:
            final_project_id = None
            credentials = None
            
            if service_account_file:
                self._log("Using provided service account file for authentication.")
                if not os.path.exists(service_account_file):
                    raise FileNotFoundError(f"Service account file not found: {service_account_file}")
                
                # Set the environment variable for Application Default Credentials
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_file
                final_project_id = self._get_project_id(service_account_file)
                credentials = service_account.Credentials.from_service_account_file(service_account_file)
            else:
                self._log("Using Application Default Credentials (e.g., gcloud auth).")
                if not user_project_id:
                    self._log("ERROR: Project ID is required when not using a service account file.")
                    return
                final_project_id = user_project_id

            self._log(f"Project ID: {final_project_id}")
            self._log("Initializing Vertex AI...")
            aiplatform.init(project=final_project_id, location=location, credentials=credentials)
            
            self._log("Initializing Generative AI Client...")
            client = genai.Client(vertexai=True, project=final_project_id, location=location)

            self._log(f"🎬 Generating video for prompt: '{prompt}'")
            
            operation = client.models.generate_videos(
                model=model,
                prompt=prompt,
                config=GenerateVideosConfig(
                    aspect_ratio=aspect_ratio,
                    number_of_videos=num_videos,
                ),
            )

            self._log("⏳ Operation started! Waiting for completion...")
            while not operation.done:
                time.sleep(15)
                operation = client.operations.get(operation)
                self._log("⏳ Still generating...")

            self._log("✅ Video generation completed!")
            
            if hasattr(operation, 'error') and operation.error:
                self._log(f"❌ Operation has error: {operation.error}")
                return
                
            if not operation.response:
                self._log("❌ Video generation completed but no response found.")
                return

            video_artifacts = []
            # Fix: videos are in operation.response, not operation.result
            generated_videos = operation.response.generated_videos if operation.response else None
            
            # Check for content filtering
            if operation.response and hasattr(operation.response, 'rai_media_filtered_count'):
                filtered_count = operation.response.rai_media_filtered_count
                if filtered_count > 0:
                    self._log(f"🚫 Content Filter: {filtered_count} video(s) were filtered by Google's content policy.")
                    if hasattr(operation.response, 'rai_media_filtered_reasons') and operation.response.rai_media_filtered_reasons:
                        for reason in operation.response.rai_media_filtered_reasons:
                            self._log(f"   Reason: {reason}")
                    self._log("💡 Tip: Try rephrasing your prompt to avoid violent, sexual, or harmful content.")
                    return
            
            if not generated_videos:
                self._log("❌ No videos found in the response.")
                return
                
            self._log(f"🎯 Generated {len(generated_videos)} video(s)")

            for i, video in enumerate(generated_videos):
                self._log(f"Processing video {i+1}...")
                video_bytes = None
                
                # Check for direct video bytes
                if hasattr(video.video, 'video_bytes') and video.video.video_bytes:
                    self._log(f"💾 Video {i+1} returned as direct bytes.")
                    video_bytes = video.video.video_bytes
                # Fallback to downloading from GCS URI
                elif hasattr(video.video, 'uri') and video.video.uri:
                    self._log(f"📹 Video {i+1} has GCS URI. Downloading...")
                    video_bytes = self._download_from_gcs(video.video.uri, final_project_id, credentials)
                
                if video_bytes:
                    filename = f"veo_video_{int(time.time())}_{i+1}.mp4"
                    self._log(f"Saving video bytes to static storage as {filename}...")
                    
                    static_files_manager = GriptapeNodes.StaticFilesManager()
                    url = static_files_manager.save_static_file(video_bytes, filename)
                    
                    url_artifact = VideoUrlArtifact(value=url, name=filename)
                    video_artifacts.append(url_artifact)
                    self._log(f"✅ Video {i+1} saved. URL: {url}")
                else:
                    self._log(f"❌ Could not retrieve video data for video {i+1}.")

            if video_artifacts:
                output_artifact = ListArtifact(video_artifacts)
                self.parameter_output_values["video_artifacts"] = output_artifact
                self._log("\n🎉 SUCCESS! All videos processed.")
            else:
                self._log("\n❌ No videos were successfully saved.")

        except FileNotFoundError as e:
            self._log(f"❌ CONFIGURATION ERROR: {e}. Please check the path to your service account file.")
        except Exception as e:
            self._log(f"❌ An unexpected error occurred: {e}")
            import traceback
            self._log(traceback.format_exc()) 


    def process(self) -> AsyncResult[None]:
            """Non-blocking entry point for Griptape engine."""
            yield lambda: self._process()