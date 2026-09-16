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
    """Grid parameter name for the nth audio clip, filling left to right, top to bottom."""
    row = (index // GRID_COLUMNS) + 1
    col = (index % GRID_COLUMNS) + 1
    return f"audio_{row}_{col}"


class AudioDisplayNode(DataNode):
    """A node that displays audio players in the UI for audio URL artifacts."""

    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
        value: Any = None,
    ) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            Parameter(
                name="audios",
                type="list",
                default_value=value or [],
                input_types=["list", "list[AudioUrlArtifact]"],
                output_type="list[AudioUrlArtifact]",
                tooltip="The list of audio clips to display",
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
                    type="AudioUrlArtifact",
                    output_type="AudioUrlArtifact",
                    tooltip=f"Audio at grid position {_cell_name(index).removeprefix('audio_').replace('_', ',')}",
                    ui_options={"hide_property": True},
                    allowed_modes={ParameterMode.OUTPUT},
                )
            )

        self._update_cell_visibility(self.get_parameter_value("audios"))

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Show only as many grid cells as there are audio clips."""
        if parameter.name == "audios":
            self._update_cell_visibility(value)
        return super().after_value_set(parameter, value)

    def process(self) -> None:
        audios = self.get_parameter_value("audios") or []

        status_lines = [f"📥 Received {len(audios)} audio clip(s)"]
        for index, audio in enumerate(audios):
            if hasattr(audio, "value"):
                status_lines.append(f"🎵 Audio {index + 1}: {audio.value} ({type(audio).__name__})")
            else:
                status_lines.append(f"⚠️ Audio {index + 1}: {audio} (no .value attribute)")
        if len(audios) > MAX_CELLS:
            status_lines.append(
                f"ℹ️ Only the first {MAX_CELLS} clips get their own output; all {len(audios)} are in 'audios'."
            )

        self.parameter_output_values["audios"] = audios

        # Every cell is assigned on every run: a cell left holding the previous run's clip would
        # keep feeding a stale artifact downstream after the list got shorter.
        for index in range(MAX_CELLS):
            self.parameter_output_values[_cell_name(index)] = audios[index] if index < len(audios) else None

        self.parameter_output_values["status"] = "\n".join(status_lines)
        self.publish_update_to_parameter("audios", audios)

    def _update_cell_visibility(self, audios: Any) -> None:
        """Reveal one grid cell per audio clip, hiding the rest."""
        count = len(audios) if isinstance(audios, list) else 0
        for index in range(MAX_CELLS):
            if index < count:
                self.show_parameter_by_name(_cell_name(index))
            else:
                self.hide_parameter_by_name(_cell_name(index))
