from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, DataNode


class AudioDisplayNode(DataNode):
    """
    A node that displays audio players in the UI for audio URL artifacts.
    """

    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
        value: Any = None,
    ) -> None:
        super().__init__(name, metadata)

        # Add parameter using your EXACT grid specification
        grid_param = Parameter(
            name="audios",
            type="list",
            default_value=value or [],
            input_types=["list", "list[AudioUrlArtifact]"],  # Accept both types
            tooltip="The list of audio clips to display",
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

        # Output parameters will be added dynamically when audios arrive

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        # Rebuild the per-clip outputs as soon as the list is set, rather than during process().
        # Parameter changes made inside process() don't propagate back to the authoritative node
        # when this library runs in Isolated mode.
        if parameter.name == "audios":
            self._sync_audio_parameters(len(value) if value else 0)
        return super().after_value_set(parameter, value)

    def process(self) -> AsyncResult[None]:
        yield lambda: self._process()

    @staticmethod
    def _grid_position(index: int) -> tuple[int, int]:
        row = (index // 2) + 1  # Row: 1, 1, 2, 2, 3, 3...
        col = (index % 2) + 1  # Col: 1, 2, 1, 2, 1, 2...
        return row, col

    def _audio_parameter_name(self, index: int) -> str:
        row, col = self._grid_position(index)
        return f"audio_{row}_{col}"

    def _sync_audio_parameters(self, audio_count: int) -> None:
        """Make the per-clip output parameters match `audio_count`.

        Only the difference is applied, so calling this when nothing has changed is a no-op. That
        matters because process() calls it too, to cover the case where the list was restored from a
        saved workflow and after_value_set never ran.
        """
        wanted = [self._audio_parameter_name(i) for i in range(audio_count)]
        existing = [param.name for param in self.parameters if param.name.startswith("audio_")]

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
                    type="AudioUrlArtifact",
                    output_type="AudioUrlArtifact",
                    tooltip=f"Audio at grid position [{row},{col}]",
                    ui_options={"hide_property": True},
                    allowed_modes={ParameterMode.OUTPUT},
                )
            )

    def _process(self):
        # Get the input audios using regular parameter method
        audios = self.get_parameter_value("audios")

        # Normally a no-op, since after_value_set has already done this. Needed for a list restored
        # from a saved workflow, which bypasses that hook.
        self._sync_audio_parameters(len(audios) if audios else 0)

        # Debug logging - this was working!
        status_msg = f"📥 Received {len(audios) if audios else 0} audio clips\n"

        if audios:
            for i, audio in enumerate(audios):
                if hasattr(audio, "value"):
                    status_msg += f"🎵 Audio {i + 1}: {audio.value}\n"
                    status_msg += f"   Type: {type(audio).__name__}\n"
                    if hasattr(audio, "mime_type"):
                        status_msg += f"   MIME: {audio.mime_type}\n"
                else:
                    status_msg += f"⚠️ Audio {i + 1}: {audio} (no .value attribute)\n"
        else:
            status_msg += "❌ No audio clips received or audios is None\n"

        # Set grid inputs and individual audio outputs
        self.parameter_output_values["audios"] = audios

        # Assign each audio to its grid position output
        for i, audio in enumerate(audios):
            self.parameter_output_values[self._audio_parameter_name(i)] = audio

        # Update status for debugging
        self.parameter_output_values["status"] = status_msg

        # Trigger UI refresh for the audios parameter
        self.publish_update_to_parameter("audios", audios)
