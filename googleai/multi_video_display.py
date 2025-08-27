from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, DataNode


class VideoDisplayNode(DataNode):
    """A node that displays video players in the UI for video URL artifacts."""

    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
        value: Any = None,
    ) -> None:
        super().__init__(name, metadata)

        # Add parameter using your EXACT grid specification
        grid_param = Parameter(
            name="videos",
            type="list",
            default_value=value or [],
            input_types=["list", "list[VideoUrlArtifact]"],  # Accept both types
            tooltip="The list of videos to display",
            ui_options={"display": "grid", "columns": 2, "pulse_on_run": True},
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(grid_param)

        # Add status parameter for debugging (input only)
        self.add_parameter(
            Parameter(
                name="status",
                type="str",
                default_value="",
                tooltip="Status and debug information",
                ui_options={"multiline": True},
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

        # Output parameters will be added dynamically when videos arrive

    def process(self) -> AsyncResult[None]:
        yield lambda: self._process()

    def _process(self):
        # Get the input videos using regular parameter method
        videos = self.get_parameter_value("videos")

        # First, dynamically add output parameters based on video count
        if videos:
            video_count = len(videos)

            # Remove any existing video output parameters first
            params_to_remove = [param for param in self.parameters if param.name.startswith("video_")]
            for param in params_to_remove:
                self.parameters.remove(param)

            # Add parameters for each video
            for i in range(video_count):
                row = (i // 2) + 1  # Row: 1, 1, 2, 2, 3, 3...
                col = (i % 2) + 1  # Col: 1, 2, 1, 2, 1, 2...
                param_name = f"video_{row}_{col}"

                self.add_parameter(
                    Parameter(
                        name=param_name,
                        type="VideoUrlArtifact",
                        output_type="VideoUrlArtifact",
                        tooltip=f"Video at grid position [{row},{col}]",
                        ui_options={"hide_property": True},
                        allowed_modes={ParameterMode.OUTPUT},
                    )
                )

        # Debug logging - this was working!
        status_msg = f"📥 Received {len(videos) if videos else 0} videos\n"

        if videos:
            for i, video in enumerate(videos):
                if hasattr(video, "value"):
                    status_msg += f"🎬 Video {i + 1}: {video.value}\n"
                    status_msg += f"   Type: {type(video).__name__}\n"
                    if hasattr(video, "mime_type"):
                        status_msg += f"   MIME: {video.mime_type}\n"
                else:
                    status_msg += f"⚠️ Video {i + 1}: {video} (no .value attribute)\n"
        else:
            status_msg += "❌ No videos received or videos is None\n"

        # Set grid inputs and individual video outputs
        self.parameter_output_values["videos"] = videos

        # Assign each video to its grid position output
        for i, video in enumerate(videos):
            row = (i // 2) + 1  # Row: 1, 1, 2, 2, 3, 3...
            col = (i % 2) + 1  # Col: 1, 2, 1, 2, 1, 2...
            param_name = f"video_{row}_{col}"
            self.parameter_output_values[param_name] = video

        # Update status for debugging
        self.parameter_output_values["status"] = status_msg

        # Trigger UI refresh for the videos parameter
        self.publish_update_to_parameter("videos", videos)
