from typing import Any
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterGroup
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.traits.options import Options

try:
    import vertexai 
    from vertexai.preview.vision_models import ImageGenerationModel
    VERTEXAI_INSTALLED = True
except ImportError:
    VERTEXAI_INSTALLED = False

class VertexAIImageGenerator(DataNode):

    def __init__(self, name: str, metadata: dict | None = None) -> None:
        super().__init__(name, metadata)    
        
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

        advanced_group.ui_options = {"hide": True}  # Hide the advanced group by default.
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



    def process(self) -> None:
        self.append_value_to_parameter("logs", "Starting image generation...\n")
        pass




