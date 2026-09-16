import base64
import logging
from typing import Any

import requests
from googleai_utils import GoogleAuthHelper, credentials_or_raise, with_extension
from griptape.artifacts import AudioUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_components.seed_parameter import SeedParameter
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options

# Attempt to import Google libraries
try:
    from google import genai

    GOOGLE_INSTALLED = True
except ImportError:
    GOOGLE_INSTALLED = False

logger = logging.getLogger("griptape_nodes_library_googleai")

# Lyria 2 and Lyria 3 are two different APIs, verified live on 2026-09-16, so the model choice
# selects a code path rather than just a string in a URL:
#
#   lyria-002       :predict on a regional endpoint -> WAV, base64 under `bytesBase64Encoded`
#   lyria-3-*       interactions.create at `global`  -> MP3, base64 under `output_audio.data`
#
# The Lyria 3 models 404 on :predict in every region, so the old predict-only path could never
# have reached them however they were spelled.
#
# lyria-3.5 and lyria-realtime-exp are deliberately absent: Vertex publishes neither, so reaching
# them would mean giving this node the AI Studio surface, which there is no way to test against.
PREDICT_MODELS = ["lyria-002"]
INTERACTION_MODELS = ["lyria-3-pro-preview", "lyria-3-clip-preview"]
MODELS = [*PREDICT_MODELS, *INTERACTION_MODELS]
DEFAULT_MODEL = PREDICT_MODELS[0]

# The predict API answers with base64 audio under this key. Google's own docs call this field
# `audioContent`, which it is not.
AUDIO_CONTENT_KEY = "bytesBase64Encoded"

# interactions is only served from the `global` location, like the Gemini Omni models.
INTERACTION_LOCATION = "global"

# Output extension per API, so the saved file matches what the model actually returns.
PREDICT_EXTENSION = ".wav"
INTERACTION_EXTENSION = ".mp3"
DEFAULT_FILENAME = f"lyria_audio{PREDICT_EXTENSION}"

# Generation is a single synchronous call; without a ceiling a hung connection hangs the node.
REQUEST_TIMEOUT_SECONDS = 300


class LyriaAudioGenerator(ControlNode):
    # Service constants for configuration
    SERVICE = "GoogleAI"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category = "Google AI"
        self.description = "Generates instrumental audio using Google's Lyria model."

        # Main Parameters
        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Describe unique instrumental music with creative specificity. Examples: 'vintage synthesizer melodies with rain sounds and distant thunder', 'acoustic guitar fingerpicking with subtle string arrangements'. Avoid generic terms like 'blues beat' or 'jazz song' to prevent copyright blocking.",
                multiline=True,
                placeholder_text="vintage synthesizer melodies with rain sounds and distant thunder",
                allow_output=False,
            )
        )

        self.add_parameter(
            ParameterString(
                name="negative_prompt",
                tooltip="Optional: Describe what to exclude from the generated audio (e.g., 'vocals, percussion, fast tempo'). This can help avoid recitation blocks by steering away from copyrighted patterns.",
                multiline=True,
                placeholder_text="vocals, percussion, fast tempo",
                allow_output=False,
            )
        )

        # No Options trait here: ModelAccessComponent installs its own, plus the license
        # decoration and the legacy-value migration.
        model_parameter = ParameterString(
            name="model",
            tooltip=(
                "The Lyria model to use. lyria-002 is generally available and returns a 30-second "
                "48kHz WAV. The Lyria 3 models are preview, return MP3, ignore the seed, and need "
                "your Google Cloud project to be allowlisted for them, otherwise they answer 404."
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

        # Seed parameter component
        self._seed_parameter = SeedParameter(self)
        self._seed_parameter.add_input_parameters()

        self.add_parameter(
            Parameter(
                name="location",
                type="str",
                tooltip=(
                    "Google Cloud location for the generation job. Only applies to lyria-002; the "
                    "Lyria 3 models are served from 'global' only."
                ),
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
                tooltip="Generated audio artifact. WAV from lyria-002, MP3 from the Lyria 3 models.",
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

        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename=DEFAULT_FILENAME)
        self._output_file.add_parameter()

        self._sync_to_model(self.get_parameter_value("model") or DEFAULT_MODEL)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Handle parameter value changes."""
        self._seed_parameter.after_value_set(parameter, value)
        self._model_access.on_value_set(parameter, value)
        if parameter.name == "model":
            self._sync_to_model(value)
        return super().after_value_set(parameter, value)

    def _sync_to_model(self, model: str) -> None:
        """Point the node's surface at whichever API the chosen model uses.

        Lyria 3 returns MP3 from a fixed location and ignores the seed, so leaving a `.wav`
        filename or a visible region would each promise something the run cannot deliver.
        """
        uses_interactions = model in INTERACTION_MODELS
        extension = INTERACTION_EXTENSION if uses_interactions else PREDICT_EXTENSION
        self._retarget_output_extension(extension)

        for name in ("location", *self._seed_parameter_names()):
            if uses_interactions:
                self.hide_parameter_by_name(name)
            else:
                self.show_parameter_by_name(name)

    def _seed_parameter_names(self) -> tuple[str, ...]:
        """The seed component's parameter names that exist on this node."""
        return tuple(name for name in ("seed", "randomize_seed") if self.get_parameter_by_name(name) is not None)

    def _retarget_output_extension(self, extension: str) -> None:
        """Swap the output filename's extension, keeping whatever base name the artist chose."""
        current = self.get_parameter_value("output_file")
        if not isinstance(current, str) or not current:
            current = DEFAULT_FILENAME
        updated = with_extension(current, extension)
        if updated != current:
            self.set_parameter_value("output_file", updated)

    def _log(self, message: str):
        """Append a message to the logs output parameter."""
        logger.info(message)
        self.append_value_to_parameter("logs", message + "\n")

    def _clear_audio_output(self) -> None:
        """Clear the audio output but keep the logs, which explain the failure."""
        self.parameter_output_values["output"] = None

    def _generate_audio(  # noqa: PLR0913
        self,
        final_project_id: str,
        credentials: Any,
        model: str,
        prompt: str,
        negative_prompt: str,
        seed: int,
        location: str,
    ) -> None:
        """Generate audio and process result - called via yield."""
        try:
            access_token = GoogleAuthHelper.get_access_token(credentials)

            url = (
                f"https://{location}-aiplatform.googleapis.com/v1/projects/{final_project_id}"
                f"/locations/{location}/publishers/google/models/{model}:predict"
            )
            headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

            instance: dict[str, Any] = {"prompt": prompt, "seed": seed}
            if negative_prompt:
                instance["negative_prompt"] = negative_prompt

            # sample_count is fixed at 1: the API returns a single clip per request.
            payload = {"instances": [instance], "parameters": {"sample_count": 1}}

            self._log(f"🎵 Generating audio with {model} for prompt: '{prompt}'")
            if negative_prompt:
                self._log(f"🚫 Negative prompt: '{negative_prompt}'")
            self._log(f"🎲 Using seed: {seed}")

            response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            result = response.json()

            audio_data = base64.b64decode(self._read_audio_content(result))
            self._log(f"✅ Decoded {len(audio_data)} bytes of audio data")

            saved = self._output_file.build_file().write_bytes(audio_data)
            self.parameter_output_values["output"] = AudioUrlArtifact(value=saved.location, name=saved.location)
            self._log(f"✅ Audio saved. URL: {saved.location}")

        except requests.exceptions.HTTPError as e:
            self._clear_audio_output()
            api_message = self._api_error_message(e)
            self._log(f"❌ API error: {api_message}")
            if e.response is not None and e.response.status_code == 404:
                self._log(
                    f"💡 '{model}' resolves on Vertex AI but your project may not be allowlisted for "
                    "it. The Lyria 3 models are preview; lyria-002 is generally available."
                )
            if "recitation" in api_message.lower() or "blocked" in api_message.lower():
                self._log(
                    "🚫 The prompt was blocked for possible copyright similarity. Describe textures and "
                    "atmosphere rather than naming a genre or artist, and try again."
                )
            msg = f"{self.name}: Lyria rejected the request. {api_message}"
            raise RuntimeError(msg) from e
        except Exception as e:
            self._clear_audio_output()
            self._log(f"❌ Audio generation failed: {e}")
            msg = f"{self.name}: Lyria audio generation failed. {e}"
            raise RuntimeError(msg) from e

    @staticmethod
    def _read_audio_content(result: dict[str, Any]) -> str:
        """Pull the base64 audio out of a predict response.

        Raises with what the response actually contained, so an API shape change reads as the
        real cause instead of a generic decode failure further down.
        """
        predictions = result.get("predictions")
        if not predictions:
            msg = f"Lyria returned no predictions. Response keys: {sorted(result)}."
            raise ValueError(msg)

        prediction = predictions[0]
        if not isinstance(prediction, dict) or AUDIO_CONTENT_KEY not in prediction:
            found = sorted(prediction) if isinstance(prediction, dict) else type(prediction).__name__
            msg = f"Lyria returned no '{AUDIO_CONTENT_KEY}' in its prediction. Found: {found}."
            raise ValueError(msg)

        return prediction[AUDIO_CONTENT_KEY]

    @staticmethod
    def _api_error_message(error: requests.exceptions.HTTPError) -> str:
        """The message Google put in the error body, or the raw error if it carried none."""
        try:
            return error.response.json().get("error", {}).get("message", str(error))
        except (ValueError, AttributeError):
            return str(error)

    def process(self) -> AsyncResult[None]:
        self._model_access.raise_if_selection_denied()
        yield lambda: self._process()

    def validate_before_node_run(self) -> list[Exception] | None:
        """Reject a run that cannot possibly produce audio."""
        exceptions: list[Exception] = []

        if not self.get_parameter_value("prompt"):
            exceptions.append(ValueError(f"{self.name}: a prompt is required."))
        model = self.get_parameter_value("model") or DEFAULT_MODEL
        if model in INTERACTION_MODELS and not GOOGLE_INSTALLED:
            exceptions.append(
                ImportError(
                    f"{self.name}: '{model}' needs 'google-genai', which is not installed. Add it "
                    "to this library's dependencies, or use lyria-002."
                )
            )

        return exceptions or None

    def _process(self):
        # Get input values
        model = self.get_parameter_value("model") or DEFAULT_MODEL
        prompt = self.get_parameter_value("prompt")
        negative_prompt = self.get_parameter_value("negative_prompt")
        location = self.get_parameter_value("location")
        uses_interactions = model in INTERACTION_MODELS

        # The output extension is re-derived here rather than trusted from the last model change:
        # `ProjectFileParameter` resets the filename to its own default when an upstream
        # destination is disconnected, which would otherwise leave `.wav` on a Lyria 3 run.
        self._retarget_output_extension(INTERACTION_EXTENSION if uses_interactions else PREDICT_EXTENSION)

        # Only the predict path takes a seed. `preprocess` randomizes and republishes it, so
        # running it for Lyria 3 would churn a hidden value the model ignores.
        seed = 0
        if not uses_interactions:
            self._seed_parameter.preprocess()
            seed = self._seed_parameter.get_seed()

        credentials, final_project_id = credentials_or_raise(
            self.name,
            log_func=self._log,
            on_failure=self._clear_audio_output,
        )
        self._log(f"Project ID: {final_project_id}")

        if uses_interactions:
            self._generate_audio_via_interactions(final_project_id, credentials, model, prompt, negative_prompt)
        else:
            self._generate_audio(final_project_id, credentials, model, prompt, negative_prompt, seed, location)

    def _generate_audio_via_interactions(  # noqa: PLR0913
        self,
        project_id: str,
        credentials: Any,
        model: str,
        prompt: str,
        negative_prompt: str,
    ) -> None:
        """Generate with a Lyria 3 model through the interactions API.

        Runs in the foreground: these models reject `background=True`, so there is no operation to
        poll. `response_format` is left off deliberately, which yields MP3 -- asking for audio
        explicitly demands a `bit_rate` and then rejects every value offered for it.
        """
        client = genai.Client(vertexai=True, project=project_id, location=INTERACTION_LOCATION, credentials=credentials)

        generation_config: dict[str, Any] = {}
        if negative_prompt:
            generation_config["audio_config"] = {"negative_prompt": negative_prompt}

        self._log(f"🎵 Generating audio with {model} for prompt: '{prompt}'")
        if negative_prompt:
            self._log(f"🚫 Negative prompt: '{negative_prompt}'")
        self._log("ℹ️ Lyria 3 ignores the seed and returns MP3.")

        try:
            kwargs: dict[str, Any] = {"model": model, "input": prompt}
            if generation_config:
                kwargs["generation_config"] = generation_config
            interaction = client.interactions.create(**kwargs)
        except Exception as e:
            self._clear_audio_output()
            self._log(f"❌ Audio generation failed: {e}")
            if "not found" in str(e).lower() or "404" in str(e):
                self._log(
                    f"💡 '{model}' is a preview model. Your Google Cloud project has to be "
                    "allowlisted for it; lyria-002 is generally available."
                )
            msg = f"{self.name}: Lyria audio generation failed. {e}"
            raise RuntimeError(msg) from e

        try:
            audio_data = base64.b64decode(self._read_interaction_audio(interaction))
            self._log(f"✅ Decoded {len(audio_data)} bytes of audio data")

            saved = self._output_file.build_file().write_bytes(audio_data)
            self.parameter_output_values["output"] = AudioUrlArtifact(value=saved.location, name=saved.location)
            self._log(f"✅ Audio saved. URL: {saved.location}")
        except Exception as e:
            self._clear_audio_output()
            self._log(f"❌ Audio generation failed: {e}")
            msg = f"{self.name}: Lyria audio generation failed. {e}"
            raise RuntimeError(msg) from e

    @staticmethod
    def _read_interaction_audio(interaction: Any) -> str:
        """Pull the base64 audio out of a completed interaction.

        Prefers the `output_audio` accessor and falls back to scanning `steps[].content[]`, which
        is where the same payload also appears.
        """
        status = getattr(interaction, "status", None)
        if status != "completed":
            msg = f"Lyria interaction ended with status '{status}' instead of completing."
            raise ValueError(msg)

        output_audio = getattr(interaction, "output_audio", None)
        if output_audio is not None and getattr(output_audio, "data", None):
            return output_audio.data

        for step in getattr(interaction, "steps", None) or []:
            for content in getattr(step, "content", None) or []:
                if getattr(content, "type", None) == "audio" and getattr(content, "data", None):
                    return content.data

        msg = "Lyria interaction completed but carried no audio content."
        raise ValueError(msg)
