# Griptape Node Development Guide v2

## Introduction
Griptape Nodes are modular workflow components. Inherit from BaseNode subclasses like DataNode for data tasks or ControlNode for flow control.

## Core Concepts
- **Base Classes**: DataNode (data processing), ControlNode (flow with exec_in/out), StartNode (workflow start), EndNode (termination).
- **Parameters**: Define inputs/outputs/properties via Parameter class.
- **process()**: Core logic; set outputs in self.parameter_output_values.
- **States**: UNRESOLVED, RESOLVING, RESOLVED.
- **Connections**: Managed via callbacks.
- **Events**: Use on_griptape_event for reactions.

## Setting Up
Install griptape-nodes. Use virtualenv. Structure: simple folders, import from griptape_nodes.exe_types.* and griptape_nodes_library.utils.*.

## Creating a Node
Inherit from appropriate base. Set self.category, self.description in __init__.

```python
from typing import Any
from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import DataNode

class MyNode(DataNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "Category"
        self.description = "Description"
        self.add_parameter(Parameter(name="input", input_types=["str"], type="str", tooltip="Input"))
        self.add_parameter(Parameter(name="output", output_type="str", tooltip="Output"))

    def process(self) -> None:
        val = self.get_parameter_value("input").upper()
        self.parameter_output_values["output"] = val
```

## Parameters
Instance of Parameter. All attributes:
- name: str, unique, no whitespace.
- tooltip: str or list[dict] for UI.
- default_value: Any.
- type: str e.g. "str", "list[str]", ParameterTypeBuiltin.STR.value.
- input_types: list[str] for incoming.
- output_type: str for outgoing.
- allowed_modes: set[ParameterMode] {INPUT, OUTPUT, PROPERTY}.
- ui_options: dict with keys: multiline(bool), hide(bool), placeholder_text(str), is_full_width(bool), className(str), display_name(str), markdown(bool), pulse_on_run(bool), expander(bool), clickable_file_browser(bool), compare(bool), slider(dict{min_val:int|float, max_val:int|float, step:int|float}), data(any).
- converters: list[Callable[[Any], Any]].
- validators: list[Callable[[Parameter, Any], None]] raise if invalid.
- settable: bool (default True).
- user_defined: bool (default False).
- parent_container_name: str|None.

Traits: Add via add_trait. Known: Options(choices=list[str]|list[tuple[str,Any]]), Slider(min_val:int|float, max_val:int|float), Button(button_type:str e.g. "save","open","action").

Containers: ParameterList, ParameterDictionary, ParameterGroup (for UI grouping).

**ParameterList Pattern:** For parameters that accept multiple inputs of the same type:
- Use `ParameterList` instead of `Parameter`
- Set `input_types` to both single and list types: `["Type", "list[Type]"]`
- Use `get_parameter_list_value()` to retrieve values (always returns a list)
- Allows users to connect multiple individual inputs OR a single list input
- UI shows multiple connection points for flexible workflow design

```python
# Multiple tools input - users can connect individual tools or tool lists
self.add_parameter(
    ParameterList(
        name="tools",
        input_types=["Tool", "list[Tool]"],
        default_value=[],
        tooltip="Connect individual tools or a list of tools",
        allowed_modes={ParameterMode.INPUT},
    )
)

# Retrieve in process method
tools = self.get_parameter_list_value("tools")  # Always returns list
for tool in tools:
    # Process each tool
```

Passthrough: Single param with INPUT+OUTPUT modes.

### Common Parameter Patterns

**Search Input with Placeholder:**
```python
Parameter(
    name="search_query",
    input_types=["str"],
    type="str",
    tooltip="Search term to find models",
    allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
    ui_options={"placeholder_text": "e.g., llama, bert, stable-diffusion"}
)
```

**Full-Width List Output:**
```python
Parameter(
    name="results",
    output_type="list[dict]",
    type="list[dict]",
    tooltip="Search results with full information",
    allowed_modes={ParameterMode.OUTPUT},
    ui_options={"is_full_width": True}
)
```

**Multiline Text Input:**
```python
Parameter(
    name="prompt",
    input_types=["str"],
    type="str",
    tooltip="Description of desired output",
    allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
    ui_options={"multiline": True, "placeholder_text": "Describe what you want..."}
)
```

**File Upload with Browser:**
```python
Parameter(
    name="image",
    input_types=["ImageArtifact", "ImageUrlArtifact", "str"],
    type="ImageArtifact",
    tooltip="Input image file",
    allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
    ui_options={"clickable_file_browser": True}
)
```

## Lifecycle Callbacks
All overridable: allow_incoming_connection, allow_outgoing_connection (return bool), after_incoming_connection, after_outgoing_connection, after_incoming_connection_removed, after_outgoing_connection_removed, before_value_set (return modified value), after_value_set, validate_before_workflow_run, validate_before_node_run (return list[Exception]|None), on_griptape_event, initialize_spotlight, get_next_control_output (return Parameter|None).

Helpers: hide/show_parameter_by_name, append_value_to_parameter, publish_update_to_parameter.

## Best Practices
- Descriptive names/tooltips.
- Robust error handling/validators.
- Single responsibility.
- Use get_config_value for keys.
- Optional deps via local imports.
- Idempotent process.

### Environment Variables & Configuration
Use `get_config_value(service="ServiceName", value="ENV_VAR_NAME")` to access environment variables. Define these in your library's settings section:

```python
# In your node
SERVICE = "Huggingface"
API_KEY_ENV_VAR = "HUGGINGFACE_HUB_ACCESS_TOKEN"

# Get the token
token = self.get_config_value(service=SERVICE, value=API_KEY_ENV_VAR)
```

### Optional Dependencies
Import optional libraries within methods to handle missing dependencies gracefully:

```python
def process(self) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        error_msg = "huggingface_hub library not installed. Please add 'huggingface_hub' to your library dependencies."
        self.parameter_output_values["output"] = None
        raise ImportError(error_msg)
```

### Error Handling Pattern
Set output values to safe defaults before raising exceptions:

```python
try:
    # Your processing logic
    result = some_operation()
    self.parameter_output_values["output"] = result
except Exception as e:
    # Set safe defaults
    self.parameter_output_values["output"] = []
    self.parameter_output_values["count"] = 0
    raise Exception(f"Error in processing: {str(e)}")
```

## Advanced Topics
### Agentic Nodes
Inherit ControlNode. Manage Agent instance. Params: agent_in/out (dict), prompt, model_config, tools (list), rulesets. Use run_stream for events, append_value_to_parameter for streaming.

### Abstract Bases
For node families: common params, abstract process, helpers.

### ParameterList for Multiple Inputs
Use `ParameterList` when your node should accept multiple inputs of the same type. This provides flexible connectivity where users can connect:
1. Multiple individual connections (each providing one item)
2. A single connection providing a list of items  
3. A combination of both approaches

**Benefits over manual list handling:**
- Cleaner UI with multiple connection points
- Automatic aggregation of inputs into a single list
- Follows Griptape design patterns
- More intuitive user experience

**Example - Image Processing Node:**
```python
# Instead of Parameter with manual list handling
self.add_parameter(
    ParameterList(
        name="images",
        input_types=["ImageUrlArtifact", "list[ImageUrlArtifact]"],
        default_value=[],
        tooltip="Connect individual images or a list of images",
        allowed_modes={ParameterMode.INPUT},
    )
)

def process(self) -> None:
    # Always returns a list, regardless of how inputs were connected
    images = self.get_parameter_list_value("images")
    
    for i, image in enumerate(images):
        # Process each image
        result = process_image(image)
        self.append_value_to_parameter("results", result)
```

**UI Behavior:**
- Shows multiple connection dots for the parameter
- Users can drag multiple connections to the same parameter
- Automatically combines all connected inputs into one list

### Dynamic Params
Add/remove in callbacks. Update traits/ui_options.

### UI Feedback
Dedicated param for status, toggle hide.

### Caching
ClassVar dict for shared resources e.g. models.

### Hubs (e.g. HF)
Params for repo_id/revision/filename. Use hf_hub_download.

**Gated Model Detection:**
```python
# In search results, detect gated models
is_gated = getattr(model, 'gated', False)
model_dict['gated'] = is_gated

# In download node, check access before download
model_info = api.model_info(repo_id)
if getattr(model_info, 'gated', False):
    self.publish_update_to_parameter("status", "🔒 GATED MODEL - May require approval")
```

**ParameterMessage for External Links:**
```python
from griptape_nodes.exe_types.core_types import ParameterMessage

# Add clickable link to external resources
ParameterMessage(
    name="model_card_link",
    title="Model Card", 
    variant="info",
    value="View model documentation on HuggingFace Hub",
    button_link=f"https://huggingface.co/{model_id}",
    button_text="View on HuggingFace"
)
```

### Custom Artifacts
Inherit BaseArtifact, override methods.

### Imports
From griptape_nodes.exe_types.*, griptape_nodes_library.utils.*. Avoid deep relatives.

### Enumerations
- NodeResolutionState: UNRESOLVED, RESOLVING, RESOLVED.
- ParameterMode: INPUT, OUTPUT, PROPERTY.
- ParameterTypeBuiltin: STR("str"), BOOL("bool"), INT("int"), FLOAT("float"), ANY("any"), NONE("none"), CONTROL_TYPE("parametercontroltype"), ALL("all"). 

### Agentic Node Example
```python
from griptape.structures import Agent
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode

class MyAgentNode(ControlNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_parameter(Parameter(name="agent_in", input_types=["Agent"], type="Agent", allowed_modes={ParameterMode.INPUT}))
        self.add_parameter(Parameter(name="prompt", input_types=["str"], type="str", allowed_modes={ParameterMode.INPUT}))
        self.add_parameter(Parameter(name="output", output_type="str", allowed_modes={ParameterMode.OUTPUT}))
        self.add_parameter(Parameter(name="agent_out", output_type="Agent", allowed_modes={ParameterMode.OUTPUT}))

    def process(self) -> None:
        agent_state = self.get_parameter_value("agent_in")
        agent = Agent.from_dict(agent_state) if agent_state else Agent()
        prompt = self.get_parameter_value("prompt")
        response = agent.run(prompt).output.value
        self.parameter_output_values["output"] = response
        self.parameter_output_values["agent_out"] = agent.to_dict()
```

### Caching Example
```python
from typing import ClassVar, Any
from griptape_nodes.exe_types.node_types import DataNode

class CachedModelNode(DataNode):
    _cache: ClassVar[dict[str, Any]] = {}

    def get_model(self, model_id: str) -> Any:
        if model_id not in self._cache:
            self._cache[model_id] = load_model(model_id)  # expensive load
        return self._cache[model_id]
``` 

## Creating Node Libraries
Bundle nodes into a library for sharing. Create griptape_nodes_library.json in library root:

```json
{
  "name": "Library Name",
  "library_schema_version": "0.1.0",
  "settings": [{"description": "Env vars", "category": "nodes.Lib", "contents": {"API_KEY": "$MY_API_KEY"}}],
  "metadata": {
    "author": "Author",
    "description": "Desc",
    "library_version": "1.0",
    "engine_version": "0.41.0",
    "tags": ["Tag"],
    "dependencies": {"pip_dependencies": ["pkg"], "pip_install_flags": ["--flag"]}
  },
  "categories": [{"cat": {"title": "Cat", "description": "Desc", "color": "border-color", "icon": "Icon"}}],
  "nodes": [{"class_name": "NodeClass", "file_path": "path/to/node.py", "metadata": {"category": "cat", "description": "Desc", "display_name": "Name", "icon": "icon", "group": "group"}}],
  "workflows": ["path/to/workflow.py"],
  "is_default_library": false
}
```

- settings: Define env vars for configs.
- metadata.dependencies: PIP packages/flags installed on library load.
- categories: Group nodes in UI.
- nodes: List classes, paths, metadata.
- workflows: Template workflows.

Structure: Flat dirs. Engine registers/loads libraries, manages deps automatically. 