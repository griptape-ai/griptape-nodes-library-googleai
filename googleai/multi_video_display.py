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

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        # Rebuild the per-video outputs as soon as the list is set, rather than during process().
        # Parameter changes made inside process() don't propagate back to the authoritative node
        # when this library runs in Isolated mode.
        if parameter.name == "videos":
            self._sync_video_parameters(len(value) if value else 0)
        return super().after_value_set(parameter, value)

    def process(self) -> AsyncResult[None]:
        yield lambda: self._process()

    @staticmethod
    def _grid_position(index: int) -> tuple[int, int]:
        row = (index // 2) + 1  # Row: 1, 1, 2, 2, 3, 3...
        col = (index % 2) + 1  # Col: 1, 2, 1, 2, 1, 2...
        return row, col

    def _video_parameter_name(self, index: int) -> str:
        row, col = self._grid_position(index)
        return f"video_{row}_{col}"

    def _sync_video_parameters(self, video_count: int) -> None:
        """Make the per-video output parameters match `video_count`.

        Only the difference is applied, so calling this when nothing has changed is a no-op. That
        matters because process() calls it too, to cover the case where the list was restored from a
        saved workflow and after_value_set never ran.
        """
        wanted = [self._video_parameter_name(i) for i in range(video_count)]
        existing = [param.name for param in self.parameters if param.name.startswith("video_")]

        for name in existing:
            if name not in wanted:
                self.remove_parameter_element_by_name(name)

        for index, name in enumerate(wanted):
            if name in existing:
                continue
            row, col = self._grid_position(index)
            self.add_parameter(
                Parameter(
                    name=name,
                    type="VideoUrlArtifact",
                    output_type="VideoUrlArtifact",
                    tooltip=f"Video at grid position [{row},{col}]",
                    ui_options={"hide_property": True},
                    allowed_modes={ParameterMode.OUTPUT},
                )
            )

    def _process(self):
        # Get the input videos using regular parameter method
        videos = self.get_parameter_value("videos")

        # Normally a no-op, since after_value_set has already done this. Needed for a list restored
        # from a saved workflow, which bypasses that hook.
        self._sync_video_parameters(len(videos) if videos else 0)

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
            self.parameter_output_values[self._video_parameter_name(i)] = video

        # Update status for debugging
        self.parameter_output_values["status"] = status_msg

        # Trigger UI refresh for the videos parameter
        self.publish_update_to_parameter("videos", videos)
