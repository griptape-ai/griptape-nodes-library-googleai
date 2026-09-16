from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING, Any

from _image_migration import (
    IMAGEN_SOURCE,
    NANO_BANANA_2_TARGET,
    NANO_BANANA_PRO_TARGET,
    MigrationTarget,
    migrate_image_node,
)
from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import (
    NodeMessageResult,
    Parameter,
    ParameterGroup,
    ParameterMessage,
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_components.seed_parameter import SeedParameter
from griptape_nodes.exe_types.param_types.parameter_button import ParameterButton
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

try:
    from google import genai
    from google.genai import types

    GOOGLE_INSTALLED = True
except ImportError:
    GOOGLE_INSTALLED = False

from googleai_utils import CREDENTIALS_HELP, GoogleAuthHelper

if TYPE_CHECKING:
    from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload

logger = logging.getLogger("griptape_nodes_library_googleai")

# Google retired the Imagen 4.0 endpoints on this date and the Imagen 3.0 endpoints on
# 2025-11-10. Every id below now answers 404 on Vertex AI.
# https://ai.google.dev/gemini-api/docs/deprecations
IMAGEN_4_RETIREMENT_DATE = date(2026, 8, 17)
# Spelled-out month, so no reader has to guess whether 08-17 is day-month or month-day.
IMAGEN_4_RETIREMENT_DATE_TEXT = IMAGEN_4_RETIREMENT_DATE.strftime("%d %B %Y")

RETIREMENT_MESSAGE = (
    f"Google retired the Imagen 4.0 models on {IMAGEN_4_RETIREMENT_DATE_TEXT} and the Imagen 3.0 "
    "models on 10 November 2025. Every model this node offers now returns 404, so it cannot "
    "generate an image under any configuration.\n\n"
    "Use one of the buttons below to migrate to a still-supported image generation node. Your "
    "prompt, aspect ratio, connections, and canvas position carry over, and this node is removed."
)

# Kept so saved workflows still resolve the value stored in their `model` parameter. None of
# these ids resolve at Google any more; the dropdown exists to be read, not to be run.
MODELS = [
    "imagen-4.0-generate-001",
    "imagen-4.0-fast-generate-001",
    "imagen-4.0-ultra-generate-001",
    "imagen-3.0-generate-002",
    "imagen-3.0-generate-001",
    "imagen-3.0-fast-generate-001",
    "imagen-3.0-capability-001",
]


class VertexAIImageGenerator(ControlNode):
    """Deprecated placeholder for Imagen image generation.

    Google has removed every Imagen endpoint, so no configuration of this node can succeed. It
    keeps its full parameter surface anyway: saved workflows set these parameters by name on
    load, so dropping them would break loading for the whole workflow rather than just this node.

    Submission is left to fail against the provider rather than being refused here, so what the
    artist sees is the real response. The deprecation message and the two migrate buttons are the
    part that has to be explained up front; the buttons rebuild the node as an image node that
    still works, carrying over values and connections. See `_image_migration` for the mappings.
    """

    # Service constants for configuration
    SERVICE = "GoogleAI"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.description = (
            f"Deprecated: Google retired Imagen on {IMAGEN_4_RETIREMENT_DATE_TEXT}. Migrate to another image node."
        )

        # Added first so the deprecation and its remedy are the first things on the node,
        # ahead of the settings that no longer reach a live model.
        self.add_node_element(
            ParameterMessage(
                name="retirement_message",
                title="Imagen is retired and this node cannot generate images",
                value=RETIREMENT_MESSAGE,
                variant="error",
            )
        )
        self.add_parameter(
            ParameterButton(
                name="migrate_to_nano_banana_2",
                label=f"Migrate to {NANO_BANANA_2_TARGET.display_name}",
                icon="replace",
                variant="default",
                full_width=True,
                tooltip="Gemini 3.1 Flash Image. Closest match: fast, general-purpose image generation.",
                on_click=self._on_migrate_to_nano_banana_2_clicked,
            )
        )
        self.add_parameter(
            ParameterButton(
                name="migrate_to_nano_banana_pro",
                label=f"Migrate to {NANO_BANANA_PRO_TARGET.display_name}",
                icon="replace",
                variant="default",
                full_width=True,
                tooltip="Gemini 3 Pro Image. Higher quality and up to 4K output, at a higher cost.",
                on_click=self._on_migrate_to_nano_banana_pro_clicked,
            )
        )

        # Main Parameters - matching text-to-video node order
        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="The text prompt for the image.",
                multiline=True,
                placeholder_text="The text prompt for the image.",
                allow_output=True,
            )
        )

        self.add_parameter(
            ParameterString(
                name="negative_prompt",
                tooltip="Optional. A description of what to discourage in the generated images. Not supported by imagen-3.0-generate-002 and newer models.",
                multiline=False,
                placeholder_text="Optional negative prompt",
                allow_output=False,
            )
        )

        self.add_parameter(
            ParameterString(
                name="model",
                tooltip="The Imagen model to use for image generation.",
                default_value=MODELS[0],
                traits=[Options(choices=MODELS)],
                allow_output=False,
            )
        )

        self.add_parameter(
            ParameterString(
                name="aspect_ratio",
                tooltip="Optional. The aspect ratio for the image.",
                default_value="1:1",
                traits={Options(choices=["1:1", "16:9", "9:16", "4:3", "3:4"])},
                allow_output=False,
            )
        )

        # Seed parameter component
        self._seed_parameter = SeedParameter(self)
        self._seed_parameter.add_input_parameters()

        self.add_parameter(
            ParameterInt(
                name="number_of_images",
                tooltip="Required. The number of images to generate.",
                default_value=1,
                traits={Options(choices=[1, 2, 3, 4])},
                ui_options={"hide": True},
                allow_output=False,
            )
        )

        self.add_parameter(
            ParameterString(
                name="location",
                tooltip="Google Cloud location for the generation job.",
                default_value="us-central1",
                traits={
                    Options(
                        choices=["us-central1", "us-east1", "us-west1", "europe-west1", "europe-west4", "asia-east1"]
                    )
                },
                allow_output=False,
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
                traits=[
                    Options(choices=["block_low_and_above", "block_medium_and_above", "block_only_high", "block_none"])
                ],
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

        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename="imagen_image.jpeg")
        self._output_file.add_parameter()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Handle parameter value changes."""
        self._seed_parameter.after_value_set(parameter, value)
        return super().after_value_set(parameter, value)

    def _log(self, message: str):
        """Append a message to the logs output parameter."""
        logger.info(message)
        self.append_value_to_parameter("logs", message + "\n")

    def _on_migrate_to_nano_banana_2_clicked(
        self,
        button: Button,  # noqa: ARG002
        button_details: ButtonDetailsMessagePayload,  # noqa: ARG002
    ) -> NodeMessageResult:
        return self._migrate(NANO_BANANA_2_TARGET)

    def _on_migrate_to_nano_banana_pro_clicked(
        self,
        button: Button,  # noqa: ARG002
        button_details: ButtonDetailsMessagePayload,  # noqa: ARG002
    ) -> NodeMessageResult:
        return self._migrate(NANO_BANANA_PRO_TARGET)

    def _migrate(self, target: MigrationTarget) -> NodeMessageResult:
        try:
            outcome = migrate_image_node(self, target, IMAGEN_SOURCE)
        except RuntimeError as e:
            # Nothing was created or rewired on this path, so the graph is untouched.
            return NodeMessageResult(success=False, details=str(e), altered_workflow_state=False)
        # Module logger, not self._log: the node has been deleted by now, so writing to its `logs`
        # output would publish an update for something the editor has already removed.
        logger.info("Migrated '%s' to '%s' (%s)", self.name, outcome.new_node_name, outcome.display_name)
        return NodeMessageResult(success=True, details=outcome.summary())

    def _create_image_artifact(self, image_bytes: bytes) -> ImageUrlArtifact:
        """Create ImageUrlArtifact using project-aware file saving."""
        try:
            saved = self._output_file.build_file().write_bytes(image_bytes)
            return ImageUrlArtifact(value=saved.location, name=saved.location)
        except Exception as e:
            raise ValueError(f"Failed to create image artifact: {e!s}") from e

    def _generate_and_process_image(
        self,
        client,
        model,
        prompt,
        number_of_images,
        seed,
        negative_prompt,
        aspect_ratio,
        output_mime_type,
        language,
        add_watermark,
        safety_filter_level,
        person_generation,
        enhance_prompt,
    ) -> None:
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
                    enhancePrompt=enhance_prompt,
                ),
            )

            self._log("✅ Image generation completed!")

            # Process the generated images
            if hasattr(image, "generated_images") and image.generated_images:
                self._log(f"Processing {len(image.generated_images)} generated image(s)...")

                # Process the first image
                gen_image = image.generated_images[0]
                if hasattr(gen_image, "image"):
                    img_obj = gen_image.image

                    # Access image bytes using the working method
                    if hasattr(img_obj, "image_bytes"):
                        image_bytes = img_obj.image_bytes
                        self._log(f"✅ Retrieved image bytes: {len(image_bytes)} bytes")

                        # Create the image artifact
                        generated_image = self._create_image_artifact(image_bytes)
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
            self._log(f"❌ Image generation failed: {e}")
            raise

    def process(self) -> AsyncResult[None]:
        yield lambda: self._process()

    def validate_before_node_run(self) -> list[Exception] | None:
        """Reject a run that cannot possibly produce an image."""
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

    def _process(self):
        # Get input values
        prompt = self.get_parameter_value("prompt")
        model = self.get_parameter_value("model")
        number_of_images = self.get_parameter_value("number_of_images")
        self._seed_parameter.preprocess()
        seed = self._seed_parameter.get_seed()
        negative_prompt = self.get_parameter_value("negative_prompt")
        aspect_ratio = self.get_parameter_value("aspect_ratio")
        output_mime_type = self.get_parameter_value("output_mime_type")
        language = self.get_parameter_value("language")
        add_watermark = self.get_parameter_value("add_watermark")
        location = self.get_parameter_value("location")
        safety_filter_level = self.get_parameter_value("safety_filter_level")
        person_generation = self.get_parameter_value("person_generation")
        enhance_prompt = self.get_parameter_value("enhance_prompt")

        # Only the credentials lookup counts as an auth failure; a ValueError raised later in
        # the run is not a credentials problem and must not be reported as one.
        try:
            credentials, final_project_id = GoogleAuthHelper.get_credentials_and_project(
                GriptapeNodes.SecretsManager(), log_func=self._log
            )
        except ValueError as e:
            self.parameter_output_values["image"] = None
            self._log(f"❌ Configuration error: {e}")
            msg = f"{self.name}: could not authenticate to Google Cloud. {e} {CREDENTIALS_HELP}"
            raise RuntimeError(msg) from e

        try:
            self._log(f"Project ID: {final_project_id}")
            self._log("Initializing Generative AI Client...")
            client = genai.Client(vertexai=True, project=final_project_id, location=location, credentials=credentials)

            self._log("Starting image generation...\n")

            # Call the image generation method directly
            self._generate_and_process_image(
                client,
                model,
                prompt,
                number_of_images,
                seed,
                negative_prompt,
                aspect_ratio,
                output_mime_type,
                language,
                add_watermark,
                safety_filter_level,
                person_generation,
                enhance_prompt,
            )

        except Exception as e:
            self.parameter_output_values["image"] = None
            self._log(f"❌ Image generation failed: {e}")
            msg = f"{self.name}: Imagen image generation failed. {e}"
            raise RuntimeError(msg) from e
