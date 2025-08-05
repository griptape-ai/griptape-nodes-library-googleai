from typing import Any
from griptape.artifacts import AudioUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode, AsyncResult


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

    def process(self) -> AsyncResult[None]:
        yield lambda: self._process()
    

    def _process(self):
        # Get the input audios using regular parameter method
        audios = self.get_parameter_value("audios")
        
        # First, dynamically add output parameters based on audio count
        if audios:
            audio_count = len(audios)
            
            # Remove any existing audio output parameters first
            params_to_remove = [param for param in self.parameters if param.name.startswith("audio_")]
            for param in params_to_remove:
                self.parameters.remove(param)
            
            # Add parameters for each audio
            for i in range(audio_count):
                row = (i // 2) + 1  # Row: 1, 1, 2, 2, 3, 3...
                col = (i % 2) + 1   # Col: 1, 2, 1, 2, 1, 2...
                param_name = f"audio_{row}_{col}"
                
                self.add_parameter(
                    Parameter(
                        name=param_name,
                        type="AudioUrlArtifact",
                        output_type="AudioUrlArtifact",
                        tooltip=f"Audio at grid position [{row},{col}]",
                        ui_options={"hide_property": True},
                        allowed_modes={ParameterMode.OUTPUT},
                    )
                )
        
        # Debug logging - this was working!
        status_msg = f"📥 Received {len(audios) if audios else 0} audio clips\n"
        
        if audios:
            for i, audio in enumerate(audios):
                if hasattr(audio, 'value'):
                    status_msg += f"🎵 Audio {i+1}: {audio.value}\n"
                    status_msg += f"   Type: {type(audio).__name__}\n"
                    if hasattr(audio, 'mime_type'):
                        status_msg += f"   MIME: {audio.mime_type}\n"
                else:
                    status_msg += f"⚠️ Audio {i+1}: {audio} (no .value attribute)\n"
        else:
            status_msg += "❌ No audio clips received or audios is None\n"
        
        # Set grid inputs and individual audio outputs
        self.parameter_output_values["audios"] = audios
        
        # Assign each audio to its grid position output
        for i, audio in enumerate(audios):
            row = (i // 2) + 1  # Row: 1, 1, 2, 2, 3, 3...
            col = (i % 2) + 1   # Col: 1, 2, 1, 2, 1, 2...
            param_name = f"audio_{row}_{col}"
            self.parameter_output_values[param_name] = audio
            
        # Update status for debugging
        self.parameter_output_values["status"] = status_msg
        
        # Trigger UI refresh for the audios parameter
        self.publish_update_to_parameter("audios", audios)