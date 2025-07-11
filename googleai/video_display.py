from typing import Any, List
from griptape.artifacts import UrlArtifact, ListArtifact
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
    A node that dynamically displays video players in the UI for a list of video URL artifacts.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "Google AI"
        self.description = "Displays video players for a list of video URLs."

        # Input parameter to accept the list of video artifacts
        self.add_parameter(
            Parameter(
                name="video_artifacts",
                input_types=["list[VideoUrlArtifact]"],
                type="list[VideoUrlArtifact]",
                tooltip="A list of VideoUrlArtifacts pointing to videos to be displayed.",
                allowed_modes={ParameterMode.INPUT},
            )
        )

        # Internal state to keep track of dynamically created parameters
        self._dynamic_video_params: List[str] = []

    def process(self) -> None:
        # The core logic is in after_value_set for dynamic UI updates
        pass

<<<<<<< HEAD
    def after_value_set(self, parameter: Parameter, value: Any, modified_parameters_set: set[str] | None = None) -> None:
=======
    def after_value_set(self, parameter: Parameter, value: Any) -> None:
>>>>>>> 7e539c2430157884c2c242a1853aec17f638a48e
        """
        Reacts to the video_artifacts input being set, and dynamically creates
        video player parameters in the UI for each video in the list.
        """
        if parameter.name == "video_artifacts":
            # 1. Clear any existing dynamic video parameters
            for param_name in self._dynamic_video_params:
                self.remove_parameter_element_by_name(param_name)
            self._dynamic_video_params.clear()

            # 2. Process the new value - handle ListArtifact properly
            video_artifacts = []
            if isinstance(value, ListArtifact):
                video_artifacts = value.value
            elif isinstance(value, list):
                video_artifacts = value

            # 3. Create new parameters for each video artifact in the list
            if video_artifacts:
                for i, artifact in enumerate(video_artifacts):
                    if isinstance(artifact, (UrlArtifact, VideoUrlArtifact)):
                        param_name = f"video_display_{i}"
                        
                        video_param = Parameter(
                            name=param_name,
                            type="VideoUrlArtifact",
                            output_type="VideoUrlArtifact",
                            allowed_modes={ParameterMode.PROPERTY, ParameterMode.OUTPUT},
                            default_value=artifact.value,
                            tooltip=f"Video player for: {artifact.name or f'Video {i+1}'}",
                            ui_options={
                                "display_name": f"Video {i+1}",
                                "is_full_width": True,
                            },
                        )
                        
                        self.add_parameter(video_param)
                        self.parameter_output_values[param_name] = artifact
                        self._dynamic_video_params.append(param_name)
<<<<<<< HEAD
                        if modified_parameters_set is not None:
                            modified_parameters_set.add(param_name)
        
 
=======
        
        return super().after_value_set(parameter, value) 
>>>>>>> 7e539c2430157884c2c242a1853aec17f638a48e
