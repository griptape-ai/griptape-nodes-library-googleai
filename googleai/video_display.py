from typing import Any, List
from griptape.artifacts import UrlArtifact, ListArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode


class VideoDisplayNode(DataNode):
    """
    A node that dynamically displays video players in the UI for a list of video URL artifacts.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category = "Google AI"
        self.description = "Displays video players for a list of video URLs."

        # Input parameter to accept the list of video artifacts
        self.video_artifacts_param = Parameter(
            name="video_artifacts",
            type="list[UrlArtifact]",
            tooltip="A list of UrlArtifacts pointing to videos to be displayed.",
        )
        self.add_parameter(self.video_artifacts_param)

        # Internal state to keep track of dynamically created parameters
        self._dynamic_video_params: List[str] = []

    def process(self) -> None:
        # The core logic is in after_value_set, so process can be empty.
        # This node is for display purposes in the UI.
        pass

    def after_value_set(self, parameter: Parameter, value: Any, modified_parameters_set: set[str]) -> None:
        """
        Reacts to the video_artifacts input being set, and dynamically creates
        video player parameters in the UI for each video in the list.
        """
        if parameter.name == self.video_artifacts_param.name:
            # 1. Clear any existing dynamic video parameters
            for param_name in self._dynamic_video_params:
                self.remove_parameter_element_by_name(param_name)
            self._dynamic_video_params.clear()

            # 2. Process the new value
            video_artifacts = []
            if isinstance(value, (ListArtifact, list)):
                value_list = value.value if isinstance(value, ListArtifact) else value
                video_artifacts = value_list

            # 3. Create new parameters for each video artifact in the list
            if video_artifacts:
                for i, artifact in enumerate(video_artifacts):
                    if isinstance(artifact, UrlArtifact):
                        param_name = f"video_display_{i}"
                        
                        video_param = Parameter(
                            name=param_name,
                            type="VideoUrlArtifact",  # Correct type for UI rendering
                            allowed_modes={ParameterMode.PROPERTY, ParameterMode.OUTPUT}, # Display and Output
                            default_value=artifact.value,
                            tooltip=f"Video player for: {artifact.name}",
                            ui_options={
                                "display_name": f"Video {i+1}",
                                "is_full_width": True,
                            },
                        )
                        
                        self.add_parameter(video_param)
                        self.parameter_output_values[param_name] = artifact
                        self._dynamic_video_params.append(param_name)
                        modified_parameters_set.add(param_name)
        
        return super().after_value_set(parameter, value, modified_parameters_set) 