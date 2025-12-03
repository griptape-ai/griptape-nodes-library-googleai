import json
import logging
import os
import time

import requests
from griptape.artifacts import AudioUrlArtifact

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

# Attempt to import Google libraries
try:
    import google.auth
    import google.auth.transport.requests
    from google.cloud import aiplatform
    from google.oauth2 import service_account

    GOOGLE_INSTALLED = True
except ImportError:
    GOOGLE_INSTALLED = False

logger = logging.getLogger("griptape_nodes_library_googleai")


class LyriaAudioGenerator(ControlNode):
    # Service constants for configuration
    SERVICE = "GoogleAI"
    SERVICE_ACCOUNT_FILE_PATH = "GOOGLE_SERVICE_ACCOUNT_FILE_PATH"
    PROJECT_ID = "GOOGLE_CLOUD_PROJECT_ID"
    CREDENTIALS_JSON = "GOOGLE_APPLICATION_CREDENTIALS_JSON"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category = "Google AI"
        self.description = "Generates instrumental audio using Google's Lyria model."

        # Main Parameters
        self.add_parameter(
            Parameter(
                name="prompt",
                type="str",
                tooltip="Describe unique instrumental music with creative specificity. Examples: 'vintage synthesizer melodies with rain sounds and distant thunder', 'acoustic guitar fingerpicking with subtle string arrangements'. Avoid generic terms like 'blues beat' or 'jazz song' to prevent copyright blocking.",
                ui_options={
                    "multiline": True,
                    "placeholder_text": "vintage synthesizer melodies with rain sounds and distant thunder",
                },
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="negative_prompt",
                type="str",
                tooltip="Optional: Describe what to exclude from the generated audio (e.g., 'vocals, percussion, fast tempo'). This can help avoid recitation blocks by steering away from copyrighted patterns.",
                ui_options={"multiline": True, "placeholder_text": "vocals, percussion, fast tempo"},
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="use_seed",
                type="bool",
                tooltip="Use a seed for deterministic generation.",
                default_value=False,
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="seed",
                type="int",
                tooltip="Seed for deterministic generation (only used when 'Use seed' is checked).",
                default_value=12345,
                ui_options={"hide_when": {"use_seed": False}},
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name="location",
                type="str",
                tooltip="Google Cloud location for the generation job.",
                default_value="us-central1",
                traits=[
                    Options(
                        choices=["us-central1", "us-east1", "us-west1", "europe-west1", "europe-west4", "asia-east1"]
                    )
                ],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        # Output Parameter
        self.add_parameter(
            Parameter(
                name="output",
                tooltip="Generated audio artifact (30-second WAV clip at 48kHz)",
                output_type="AudioUrlArtifact",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        # Logs Group
        with ParameterGroup(name="Logs") as logs_group:
            Parameter(
                name="logs",
                type="str",
                tooltip="Logs from the audio generation process.",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"multiline": True, "placeholder_text": "Logs"},
            )

        logs_group.ui_options = {"hide": True}
        self.add_node_element(logs_group)

    def _log(self, message: str):
        """Append a message to the logs output parameter."""
        logger.info(message)
        self.append_value_to_parameter("logs", message + "\n")

    def _get_project_id(self, service_account_file: str) -> str:
        """Read the project_id from the service account JSON file."""
        if not os.path.exists(service_account_file):
            raise FileNotFoundError(f"Service account file not found: {service_account_file}")

        with open(service_account_file) as f:
            service_account_info = json.load(f)

        project_id = service_account_info.get("project_id")
        if not project_id:
            raise ValueError("No 'project_id' found in the service account file.")

        return project_id

    def _get_access_token(self, credentials) -> str:
        """Get access token from credentials."""
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
        return credentials.token

    def _generate_audio(self, final_project_id, credentials, prompt, negative_prompt, use_seed, seed, location) -> None:
        """Generate audio and process result - called via yield."""
        try:
            # Get access token
            access_token = self._get_access_token(credentials)

            # Build the API request
            url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{final_project_id}/locations/{location}/publishers/google/models/lyria-002:predict"

            headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

            # Build instance data
            instance = {"prompt": prompt}

            if negative_prompt:
                instance["negative_prompt"] = negative_prompt

            if use_seed:
                instance["seed"] = seed

            # Build parameters - hardcoded to 1 due to API limitation
            parameters = {"sample_count": 1}

            # Build request payload
            payload = {"instances": [instance], "parameters": parameters}

            # Debug: Log the request payload
            self._log("🔍 Request payload:")
            self._log(json.dumps(payload, indent=2))

            self._log(f"🎵 Generating audio for prompt: '{prompt}'")
            if negative_prompt:
                self._log(f"🚫 Negative prompt: '{negative_prompt}'")
            if use_seed:
                self._log(f"🎲 Using seed: {seed}")
            else:
                self._log("🎵 Generating 1 audio clip")

            # Log helpful tip for avoiding recitation blocks
            self._log("💡 TIP: If you get blocked by recitation checks, try more unique/creative prompts!")

            # Make the API request
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json()

            self._log("✅ Audio generation completed!")

            # Debug: Log the complete response structure
            self._log("🔍 Complete API response structure:")
            self._log(f"   Response keys: {list(result.keys())}")

            # Show all non-prediction fields
            for key, value in result.items():
                if key != "predictions":
                    self._log(f"   {key}: {value}")

            # Show predictions structure
            if "predictions" in result:
                predictions = result["predictions"]
                self._log(f"   predictions: array with {len(predictions)} item(s)")
                for i, pred in enumerate(predictions):
                    if isinstance(pred, dict):
                        self._log(f"     Prediction {i + 1} keys: {list(pred.keys())}")
                        for k, v in pred.items():
                            if isinstance(v, str) and len(v) > 100:
                                self._log(f"       {k}: <string with {len(v)} characters>")
                            else:
                                self._log(f"       {k}: {v}")
                    else:
                        self._log(f"     Prediction {i + 1}: {type(pred).__name__}")

            # Process the predictions
            if "predictions" not in result or not result["predictions"]:
                self._log("❌ No predictions found in response.")
                return

            predictions = result["predictions"]

            # Process the first prediction (API only generates 1 audio clip)
            prediction = predictions[0]
            self._log("🎯 Processing generated audio clip...")

            # Debug: Log prediction structure
            self._log("🔍 Prediction structure:")
            if isinstance(prediction, dict):
                self._log(f"   Keys: {list(prediction.keys())}")
                for key, value in prediction.items():
                    if isinstance(value, str) and len(value) > 100:
                        self._log(f"   {key}: <string with {len(value)} characters>")
                    else:
                        self._log(f"   {key}: {value}")
            else:
                self._log(f"   Prediction is type: {type(prediction)}")
                if isinstance(prediction, str):
                    self._log(f"   String length: {len(prediction)}")
                    if len(prediction) > 1000:
                        self._log(f"   First 100 chars: {prediction[:100]}...")
                    else:
                        self._log(f"   Content: {prediction}")
                else:
                    self._log(f"   Value: {prediction}")

            self._log("Processing audio...")

            # Try different possible field names for audio content
            audio_content = None
            found_in_field = None

            if isinstance(prediction, dict):
                # Try various field names
                field_names = ["audioContent", "audio_content", "content", "data", "audio", "prediction"]
                for field_name in field_names:
                    if field_name in prediction:
                        audio_content = prediction[field_name]
                        found_in_field = field_name
                        break

                # If no named field, try to find any string field that looks like base64
                if not audio_content:
                    for key, value in prediction.items():
                        if isinstance(value, str) and len(value) > 1000:
                            # Likely base64 audio data
                            audio_content = value
                            found_in_field = key
                            break
            elif isinstance(prediction, str):
                # Sometimes the prediction itself is the base64 string
                audio_content = prediction
                found_in_field = "direct_string"

            if not audio_content:
                self._log("❌ No audio content found in any expected field")
                return

            self._log(f"✅ Found audio content in field: {found_in_field}")

            # Decode base64 audio data
            import base64

            try:
                audio_data = base64.b64decode(audio_content)
                self._log(f"✅ Successfully decoded {len(audio_data)} bytes of audio data")
            except Exception as e:
                self._log(f"❌ Failed to decode base64 audio data: {e}")
                return

            # Generate filename
            filename = f"lyria_audio_{int(time.time())}.wav"
            self._log(f"Saving audio to static storage as {filename}...")

            # Save using StaticFilesManager
            static_files_manager = GriptapeNodes.StaticFilesManager()
            url = static_files_manager.save_static_file(audio_data, filename)

            url_artifact = AudioUrlArtifact(value=url, name=filename)
            self.parameter_output_values["output"] = url_artifact
            self._log(f"✅ Audio saved. URL: {url}")
            self._log("\n🎉 SUCCESS! Audio processed.")
            self._log("🎵 Generated 30-second instrumental WAV clip at 48kHz")

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                try:
                    error_detail = e.response.json()
                    error_message = error_detail.get("error", {}).get("message", "Bad Request")
                    self._log(f"❌ API Error: {error_message}")

                    # Check for recitation/copyright blocking issues
                    if "recitation" in error_message.lower() or "blocked" in error_message.lower():
                        self._log("\n🚫 RECITATION/COPYRIGHT BLOCK DETECTED:")
                        self._log("   This prompt was blocked due to potential copyright similarities.")
                        self._log("\n💡 SOLUTIONS TO TRY:")
                        self._log("   1. Make your prompt more unique and specific")
                        self._log(
                            "   2. Use creative genre combinations (e.g., 'psychedelic cumbia', 'hazy uk garage')"
                        )
                        self._log("   3. Focus on textures/atmosphere rather than genres")
                        self._log("   4. Add environmental sounds or unique elements")
                        self._log("\n✅ EXAMPLE PROMPTS THAT WORK:")
                        self._log("   • 'vintage synthesizer melodies with rain sounds and distant thunder'")
                        self._log("   • 'acoustic guitar fingerpicking with subtle string arrangements'")
                        self._log("   • 'ambient electronic soundscape with field recording elements'")
                        self._log("   • 'minimalist piano with reverb over soft nature sounds'")
                        self._log("\n🔄 You can also try running the same prompt again - sometimes it works on retry!")

                except:
                    self._log("❌ API Error: Bad Request (400)")
            else:
                self._log(f"❌ HTTP Error: {e}")
        except Exception as e:
            self._log(f"❌ An unexpected error occurred during audio generation: {e}")
            import traceback

            self._log(traceback.format_exc())
            raise

    def process(self) -> AsyncResult[None]:
        yield lambda: self._process()

    def _process(self):
        if not GOOGLE_INSTALLED:
            self._log(
                "ERROR: Required Google libraries are not installed. Please add 'google-auth', 'google-cloud-aiplatform' to your library's dependencies."
            )
            return

        # Get input values
        prompt = self.get_parameter_value("prompt")
        negative_prompt = self.get_parameter_value("negative_prompt")
        use_seed = self.get_parameter_value("use_seed")
        seed = self.get_parameter_value("seed")
        location = self.get_parameter_value("location")

        # Validate inputs
        if not prompt:
            self._log("ERROR: Prompt is a required input.")
            return

        try:
            final_project_id = None
            credentials = None

            # Try service account file first
            service_account_file = GriptapeNodes.SecretsManager().get_secret(f"{self.SERVICE_ACCOUNT_FILE_PATH}")

            if service_account_file and os.path.exists(service_account_file):
                self._log("🔑 Using service account file for authentication.")
                try:
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_file
                    final_project_id = self._get_project_id(service_account_file)
                    # Add required scopes for Vertex AI API
                    credentials = service_account.Credentials.from_service_account_file(
                        service_account_file, scopes=["https://www.googleapis.com/auth/cloud-platform"]
                    )
                    self._log(f"✅ Service account authentication successful for project: {final_project_id}")
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
                        "❌ GOOGLE_CLOUD_PROJECT_ID must be set in library settings when not using a service account file."
                    )

                if credentials_json:
                    try:
                        import json

                        cred_dict = json.loads(credentials_json)
                        # Add required scopes for Vertex AI API
                        credentials = service_account.Credentials.from_service_account_info(
                            cred_dict, scopes=["https://www.googleapis.com/auth/cloud-platform"]
                        )
                        self._log("✅ JSON credentials authentication successful.")
                    except Exception as e:
                        self._log(f"❌ JSON credentials authentication failed: {e}")
                        raise
                else:
                    self._log("🔑 Using Application Default Credentials (e.g., gcloud auth).")
                    # Add required scopes for Vertex AI API
                    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

                final_project_id = project_id

            self._log(f"Project ID: {final_project_id}")
            self._log("Initializing Vertex AI...")
            aiplatform.init(project=final_project_id, location=location, credentials=credentials)

            # Generate the audio
            self._generate_audio(final_project_id, credentials, prompt, negative_prompt, use_seed, seed, location)

        except ValueError as e:
            self._log(f"❌ CONFIGURATION ERROR: {e}")
            self._log("💡 Please set up Google Cloud credentials in the library settings:")
            self._log("   - GOOGLE_SERVICE_ACCOUNT_FILE_PATH (path to service account JSON)")
            self._log("   - OR GOOGLE_CLOUD_PROJECT_ID + GOOGLE_APPLICATION_CREDENTIALS_JSON")
        except Exception as e:
            self._log(f"❌ An unexpected error occurred: {e}")
            import traceback

            self._log(traceback.format_exc())
