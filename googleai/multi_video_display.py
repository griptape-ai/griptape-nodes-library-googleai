from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode

# The per-cell outputs are declared up front rather than grown to fit the incoming list, because
# adding or removing parameters from inside `process` only mutates the transient node the worker
# built for that run and never reaches the orchestrator's authoritative copy.
GRID_COLUMNS = 2
GRID_ROWS = 4
MAX_CELLS = GRID_COLUMNS * GRID_ROWS


def _cell_name(index: int) -> str:
    """Grid parameter name for the nth video, filling left to right, top to bottom."""
    row = (index // GRID_COLUMNS) + 1
    col = (index % GRID_COLUMNS) + 1
    return f"video_{row}_{col}"


class VideoDisplayNode(DataNode):
    """A node that displays video players in the UI for video URL artifacts."""

    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
        value: Any = None,
    ) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            Parameter(
                name="videos",
                type="list",
                default_value=value or [],
                input_types=["list", "list[VideoUrlArtifact]"],
                output_type="list[VideoUrlArtifact]",
                tooltip="The list of videos to display",
                ui_options={"display": "grid", "columns": GRID_COLUMNS, "pulse_on_run": True},
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="status",
                type="str",
                default_value="",
                tooltip="Status and debug information",
                ui_options={"multiline": True},
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        for index in range(MAX_CELLS):
            self.add_parameter(
                Parameter(
                    name=_cell_name(index),
                    type="VideoUrlArtifact",
                    output_type="VideoUrlArtifact",
                    tooltip=f"Video at grid position {_cell_name(index).removeprefix('video_').replace('_', ',')}",
                    ui_options={"hide_property": True},
                    allowed_modes={ParameterMode.OUTPUT},
                )
            )

        self._update_cell_visibility(self.get_parameter_value("videos"))

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Show only as many grid cells as there are videos."""
        if parameter.name == "videos":
            self._update_cell_visibility(value)
        return super().after_value_set(parameter, value)

    def process(self) -> None:
        videos = self.get_parameter_value("videos") or []

        status_lines = [f"📥 Received {len(videos)} video(s)"]
        for index, video in enumerate(videos):
            if hasattr(video, "value"):
                status_lines.append(f"🎬 Video {index + 1}: {video.value} ({type(video).__name__})")
            else:
                status_lines.append(f"⚠️ Video {index + 1}: {video} (no .value attribute)")
        if len(videos) > MAX_CELLS:
            status_lines.append(
                f"ℹ️ Only the first {MAX_CELLS} videos get their own output; all {len(videos)} are in 'videos'."
            )

        self.parameter_output_values["videos"] = videos

        # Every cell is assigned on every run: a cell left holding the previous run's video would
        # keep feeding a stale artifact downstream after the list got shorter.
        for index in range(MAX_CELLS):
            self.parameter_output_values[_cell_name(index)] = videos[index] if index < len(videos) else None

        self.parameter_output_values["status"] = "\n".join(status_lines)
        self.publish_update_to_parameter("videos", videos)

    def _update_cell_visibility(self, videos: Any) -> None:
        """Reveal one grid cell per video, hiding the rest."""
        count = len(videos) if isinstance(videos, list) else 0
        for index in range(MAX_CELLS):
            if index < count:
                self.show_parameter_by_name(_cell_name(index))
            else:
                self.hide_parameter_by_name(_cell_name(index))
