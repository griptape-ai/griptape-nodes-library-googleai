from typing import Any
from griptape.artifacts import UrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode


class VideoUrlArtifact(UrlArtifact):
    """
    Artifact that contains a URL to a video.
    """
    def __init__(self, value: str, name: str | None = None):
        super().__init__(value=value, name=name or self.__class__.__name__)


class VideoDisplayNode(DataNode):
    """
    A node that displays video players in the UI for video URL artifacts.
    """
    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
        value: Any = None,
    ) -> None:
        super().__init__(name, metadata)

        # Add parameter for the videos
        self.add_parameter(
            Parameter(
                name="videos",
                default_value=value,
                input_types=["VideoUrlArtifact", "list[VideoUrlArtifact]"],
                output_type="VideoUrlArtifact",
                type="VideoUrlArtifact", 
                tooltip="The videos to display",
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT, ParameterMode.PROPERTY},
            )
        )

    def process(self) -> None:
        # Simply output the input videos
        self.parameter_output_values["videos"] = self.parameter_values.get("videos")
