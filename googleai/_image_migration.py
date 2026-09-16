"""Rebuild a retired image-generation node as one that can still run.

Google shut down every Imagen model on 2026-08-17 (4.0) and 2025-11-10 (3.0), and removes
gemini-2.5-flash-image on 2026-10-02. The nodes built on those models stay in the library only
so saved workflows still load; this module is the escape hatch that turns one into a working
node without the artist rewiring the graph by hand.

Both retired nodes migrate to the same two survivors, so the mappings live here once rather
than once per node.
"""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from googleai_utils import with_extension
from griptape_nodes.retained_mode.events.connection_events import (
    CreateConnectionRequest,
    CreateConnectionResultSuccess,
    DeleteConnectionRequest,
    DeleteConnectionResultSuccess,
)
from griptape_nodes.retained_mode.events.node_events import (
    CreateNodeRequest,
    CreateNodeResultSuccess,
    DeleteNodeRequest,
    DeleteNodeResultSuccess,
    GetFlowForNodeRequest,
    GetFlowForNodeResultSuccess,
    GetNodeMetadataRequest,
    GetNodeMetadataResultSuccess,
)
from griptape_nodes.retained_mode.events.parameter_events import SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

if TYPE_CHECKING:
    from collections.abc import Iterable

    from griptape_nodes.exe_types.node_types import BaseNode

logger = logging.getLogger("griptape_nodes_library_googleai")

__all__ = [
    "IMAGEN_SOURCE",
    "NANO_BANANA_PRO_TARGET",
    "NANO_BANANA_SOURCE",
    "NANO_BANANA_2_TARGET",
    "MigrationOutcome",
    "MigrationSource",
    "MigrationTarget",
    "migrate_image_node",
]

# The locations both surviving nodes offer. A retired node's region that is not among these has
# no counterpart, so the migration falls back to the one region that always serves these models.
_TARGET_LOCATIONS = frozenset({"global", "us-central1", "europe-west1", "asia-southeast1"})
_FALLBACK_LOCATION = "global"

# A ParameterList row is a generated child parameter, e.g.
# `input_images_ParameterListUniqueParamID_9f1c3a...`. Reports name the owning list instead, since
# the generated suffix means nothing to the artist looking for what to reconnect.
_LIST_ROW_SUFFIX = re.compile(r"_ParameterList(?:Unique)?ParamID_[0-9a-f]+$")

# Carried from the retired node so the replacement lands where it stood. Everything else in a
# node's metadata is instance-specific and must not transfer.
_LAYOUT_METADATA_KEYS = frozenset({"position", "size"})

# Both targets write PNG regardless of what the filename says, so a carried name keeps its base
# and takes this extension rather than promising a format the bytes are not.
_TARGET_EXTENSION = ".png"


@dataclass(frozen=True)
class MigrationTarget:
    """A node type a retired image node can be rebuilt as.

    Attributes:
        node_type: Class name registered in the library, as `CreateNodeRequest` takes it.
        display_name: Target's library display name, for messages the artist reads.
    """

    node_type: str
    display_name: str


@dataclass(frozen=True)
class MigrationSource:
    """What a particular retired node can hand to a target.

    Kept separate from `MigrationTarget` because the two survivors share one parameter surface,
    so what varies between migrations is the node being left behind, not the one being created.

    Attributes:
        carried_parameters: Source parameter names worth reading. A name absent from the source
            node at runtime is skipped, so this can list parameters that only some versions had.
        parameter_renames: Source parameter name -> target parameter name, for the ones that
            carry the same meaning under a different name.
        unsupported_parameters: Source parameter name -> the note explaining why it cannot come
            across. Reported only when the artist actually set the parameter, so a node left on
            defaults migrates without a wall of irrelevant warnings.
    """

    carried_parameters: tuple[str, ...]
    parameter_renames: dict[str, str] = field(default_factory=dict)
    unsupported_parameters: dict[str, str] = field(default_factory=dict)


@dataclass
class MigrationOutcome:
    """What `migrate_image_node` managed to carry over.

    Attributes:
        new_node_name: Name the engine assigned the replacement node.
        display_name: Replacement node's library display name.
        dropped_connections: Human-readable connections that could not be recreated, because
            the target has no counterpart parameter or refused the type.
        notes: Value translations an artist would want to know about, such as a setting the
            target has no equivalent for.
    """

    new_node_name: str
    display_name: str
    dropped_connections: list[str]
    notes: list[str]

    def summary(self) -> str:
        """Render the outcome as the message shown after the button click."""
        lines = [f"Replaced this node with '{self.new_node_name}' ({self.display_name})."]
        if self.notes:
            lines.append("")
            lines.extend(f"- {note}" for note in self.notes)
        if self.dropped_connections:
            lines.append("")
            lines.append("Connections that could not be carried over -- reconnect these by hand:")
            lines.extend(f"- {dropped}" for dropped in self.dropped_connections)
        return "\n".join(lines)


NANO_BANANA_2_TARGET = MigrationTarget(
    node_type="NanaBanana2ImageGenerator",
    display_name="Nano Banana 2 Image Generator",
)

NANO_BANANA_PRO_TARGET = MigrationTarget(
    node_type="NanoBananaProImageGenerator",
    display_name="Nano Banana Pro Image Generator",
)

# Imagen took its guidance as a separate negative prompt and could sample several images per
# run; the Gemini image models do neither, so those two are reported rather than translated.
IMAGEN_SOURCE = MigrationSource(
    carried_parameters=("prompt", "aspect_ratio", "location", "output_file"),
    unsupported_parameters={
        "seed": "Seed was dropped: the Gemini image models do not accept one, so results will vary per run.",
        "output_mime_type": (
            "Output format was dropped: the Gemini image models return PNG. Rename the output file "
            "if you need a different extension."
        ),
        "negative_prompt": (
            "Negative prompt was dropped: the Gemini image models take no negative prompt. "
            "Fold what you wanted to avoid into the prompt itself."
        ),
        "number_of_images": (
            "Image count was dropped: the Gemini image models return what they return. "
            "Run the node more than once for more variations."
        ),
        "safety_filter_level": "Safety filter level was dropped: the Gemini image models do not expose it.",
        "person_generation": "Person generation setting was dropped: the Gemini image models do not expose it.",
        "add_watermark": "Watermark setting was dropped: the Gemini image models do not expose it.",
        "language": "Prompt language setting was dropped: the Gemini image models detect language themselves.",
        "enhance_prompt": "Prompt rewriting setting was dropped: the Gemini image models do not expose it.",
    },
)

NANO_BANANA_SOURCE = MigrationSource(
    carried_parameters=(
        "prompt",
        "aspect_ratio",
        "location",
        "temperature",
        "top_p",
        "auto_image_resize",
        "output_file",
    ),
    unsupported_parameters={
        # A ParameterList's value cannot be handed over: `get_parameter_value` only reads a
        # list's stored whole-list value when the list itself has an incoming connection, and
        # otherwise rebuilds it from child rows, which a migration cannot create. Writing the
        # list would therefore leave the target reading an empty list and the artist believing
        # the images came across, so they are reported as dropped instead.
        "input_images": (
            "Reference images were not carried over: reconnect or re-add them on the new node. "
            "Its 'reference_images' input takes the same images."
        ),
        "input_files": (
            "Document inputs were dropped: the target node takes reference images only. "
            "Paste the text you need into the prompt."
        ),
        "candidate_count": "Candidate count was dropped: the target node does not expose it.",
    },
)


def migrate_image_node(
    source_node: BaseNode,
    target: MigrationTarget,
    source: MigrationSource,
) -> MigrationOutcome:
    """Replace `source_node` with a `target` node holding its values and connections.

    The replacement lands at the retired node's canvas position and the retired node is
    deleted, so one click leaves a graph that runs. Connections the target cannot accept are
    reported on the outcome rather than failing the whole migration: a mostly-rewired graph the
    artist can finish beats an untouched dead one.

    Args:
        source_node: The retired image node to replace.
        target: The node type to rebuild it as.
        source: What the retired node can hand over.

    Returns:
        What was carried over, including anything the artist has to reconnect by hand.

    Raises:
        RuntimeError: If the replacement node could not be created, in which case the retired
            node and its connections are left exactly as they were.
    """
    source_name = source_node.name

    flow_result = GriptapeNodes.handle_request(GetFlowForNodeRequest(node_name=source_name))
    if not isinstance(flow_result, GetFlowForNodeResultSuccess):
        msg = (
            f"Attempted to replace '{source_name}' with a {target.display_name} node. "
            f"Failed because the flow containing '{source_name}' could not be found."
        )
        raise RuntimeError(msg)

    # Only the layout keys, deep-copied. `GetNodeMetadataRequest` hands back the node's own
    # metadata dict, and `Library.create_node` rewrites `node_type` / `library` on whatever it is
    # given, so passing the whole thing would leave the retired node claiming to be its
    # replacement and the two sharing one `position`. That is invisible while the delete succeeds
    # and corrupts the leftover node when it does not. The engine's own reset-to-defaults copies
    # the same two keys for the same reason (node_manager.on_reset_node_to_defaults_request).
    metadata = None
    metadata_result = GriptapeNodes.handle_request(GetNodeMetadataRequest(node_name=source_name))
    if isinstance(metadata_result, GetNodeMetadataResultSuccess):
        metadata = {
            key: deepcopy(value) for key, value in metadata_result.metadata.items() if key in _LAYOUT_METADATA_KEYS
        }

    # Snapshot before touching anything: deleting the retired node cascades its connections away.
    incoming, outgoing = _snapshot_connections(source_name)
    source_values = _read_source_values(source_node, source)

    create_result = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type=target.node_type,
            override_parent_flow_name=flow_result.flow_name,
            metadata=metadata,
            create_error_proxy_on_failure=False,
        )
    )
    if not isinstance(create_result, CreateNodeResultSuccess):
        msg = (
            f"Attempted to replace '{source_name}' with a {target.display_name} node. "
            f"Failed because the {target.display_name} node could not be created."
        )
        raise RuntimeError(msg)

    new_name = create_result.node_name
    source_defaults = _source_defaults(source_node, (*source.carried_parameters, *source.unsupported_parameters))
    notes = _apply_values(new_name, source_name, source_values, source_defaults, source)

    # Free the downstream input slots before reconnecting: an input parameter holds one incoming
    # connection, so the retired node has to let go before the replacement can take over. A slot
    # that refuses to release cannot be reconnected, so the two outcomes are kept apart and only
    # the released ones are offered to the replacement.
    released: list[tuple[str, str, str]] = []
    unreleased: list[tuple[str, str, str]] = []
    for connection in outgoing:
        source_param, target_node_name, target_param = connection
        result = GriptapeNodes.handle_request(
            DeleteConnectionRequest(
                source_node_name=source_name,
                source_parameter_name=source_param,
                target_node_name=target_node_name,
                target_parameter_name=target_param,
            )
        )
        if isinstance(result, DeleteConnectionResultSuccess):
            released.append(connection)
        else:
            unreleased.append(connection)

    dropped = _reconnect(new_name, source_name, incoming, released, source)
    # Name the real cause. Left to fail in `_reconnect`, these would be reported as though the
    # target had rejected the connection, sending the artist to look at the wrong node.
    dropped.extend(
        f"{source_name}.{param} -> {downstream_node}.{downstream_param} (the existing connection could not be released)"
        for param, downstream_node, downstream_param in unreleased
    )

    # The replacement is already wired in by this point, so a delete that fails leaves two nodes
    # driving the same downstream inputs. Say so rather than reporting a clean migration.
    delete_result = GriptapeNodes.handle_request(DeleteNodeRequest(node_name=source_name))
    if not isinstance(delete_result, DeleteNodeResultSuccess):
        notes.append(
            f"'{source_name}' could not be deleted and is still on the canvas, still fed by whatever "
            f"was wired into it. Its downstream connections now come from '{new_name}', so delete "
            f"'{source_name}' by hand."
        )

    return MigrationOutcome(
        new_node_name=new_name,
        display_name=target.display_name,
        dropped_connections=dropped,
        notes=notes,
    )


def _snapshot_connections(
    node_name: str,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Record the node's connections as plain names, surviving the node's deletion.

    Returns:
        Incoming as (upstream node, upstream parameter, this node's parameter), and outgoing as
        (this node's parameter, downstream node, downstream parameter).
    """
    connections = GriptapeNodes.FlowManager().get_connections()

    incoming = [
        (
            connections.connections[connection_id].source_node.name,
            connections.connections[connection_id].source_parameter.name,
            parameter_name,
        )
        for parameter_name, connection_ids in connections.incoming_index.get(node_name, {}).items()
        for connection_id in connection_ids
    ]
    outgoing = [
        (
            parameter_name,
            connections.connections[connection_id].target_node.name,
            connections.connections[connection_id].target_parameter.name,
        )
        for parameter_name, connection_ids in connections.outgoing_index.get(node_name, {}).items()
        for connection_id in connection_ids
    ]
    return incoming, outgoing


def _read_source_values(source_node: BaseNode, source: MigrationSource) -> dict[str, Any]:
    """Read the retired node's values a migration cares about, plus anything it must report.

    The unsupported parameters are read too, so `_apply_values` can tell an artist who set one
    that it is being left behind, and stay quiet for one still on its default.
    """
    values: dict[str, Any] = {}
    for name in (*source.carried_parameters, *source.unsupported_parameters):
        if source_node.get_parameter_by_name(name) is None:
            continue
        values[name] = source_node.get_parameter_value(name)
    return values


def _source_defaults(source_node: BaseNode, names: Iterable[str]) -> dict[str, Any]:
    """The declared default of each named parameter that exists on the node."""
    defaults: dict[str, Any] = {}
    for name in names:
        parameter = source_node.get_parameter_by_name(name)
        if parameter is not None:
            defaults[name] = parameter.default_value
    return defaults


def _apply_values(  # noqa: PLR0913
    new_name: str,
    source_name: str,
    source_values: dict[str, Any],
    source_defaults: dict[str, Any],
    source: MigrationSource,
) -> list[str]:
    """Set the target's parameters from the retired node's values.

    Returns:
        Notes about translations the artist would want to know about.
    """
    notes: list[str] = []
    values: dict[str, Any] = {}

    for name in source.carried_parameters:
        value = source_values.get(name)
        if value is None or value == "" or value == []:
            continue
        values[source.parameter_renames.get(name, name)] = value

    # An output filename the artist never edited names the retired node and carries its
    # extension, so let the target name its own file instead of inheriting a stale one. A
    # customized name is kept, but re-extensioned: Imagen could emit JPEG and the targets only
    # emit PNG, so carrying `hero.jpeg` across would put PNG bytes in a JPEG-named file.
    output_file = values.get("output_file")
    if output_file == source_defaults.get("output_file"):
        values.pop("output_file", None)
    elif isinstance(output_file, str):
        retargeted = with_extension(output_file, _TARGET_EXTENSION)
        if retargeted != output_file:
            values["output_file"] = retargeted
            notes.append(f"Output file renamed from '{output_file}' to '{retargeted}': the target node emits PNG.")

    # A region the artist never chose is not worth carrying: the sources default to a regional
    # endpoint and the targets default to `global`, which is where the Gemini image models are
    # actually served, so inheriting the source's default would pin the replacement to a region
    # that may 404 on its first run.
    if values.get("location") == source_defaults.get("location"):
        values.pop("location", None)

    # A region the target does not offer would be rewritten to the first dropdown entry with no
    # explanation, so translate it here and say so.
    location = values.get("location")
    if isinstance(location, str) and location not in _TARGET_LOCATIONS:
        values["location"] = _FALLBACK_LOCATION
        notes.append(f"Location changed from '{location}' to '{_FALLBACK_LOCATION}', which the target node serves.")

    for name, note in source.unsupported_parameters.items():
        if _was_set(source_values.get(name), source_defaults.get(name)):
            notes.append(note)

    for parameter_name, value in values.items():
        result = GriptapeNodes.handle_request(
            SetParameterValueRequest(node_name=new_name, parameter_name=parameter_name, value=value)
        )
        logger.debug("Migrating %s: set %s.%s -> %s", source_name, new_name, parameter_name, result)

    return notes


def _readable_parameter(name: str) -> str:
    """A parameter name an artist can find on the node, for dropped-connection reports."""
    return _LIST_ROW_SUFFIX.sub("", name)


def _was_set(value: Any, default: Any) -> bool:
    """Whether the artist changed a parameter this migration cannot carry.

    Compared against the parameter's own declared default rather than truthiness: most of the
    settings being reported here default to a non-empty value (`allow_adult`, `block_medium_and_above`,
    `enhance_prompt=True`), so a truthiness test would report every one of them on a node nobody
    touched and bury the notes that matter.
    """
    if value is None or value == "" or value == []:
        return False
    return value != default


def _reconnect(
    new_name: str,
    source_name: str,
    incoming: list[tuple[str, str, str]],
    released_outgoing: list[tuple[str, str, str]],
    source: MigrationSource,
) -> list[str]:
    """Rebuild the retired node's connections on the replacement.

    Args:
        new_name: The replacement node.
        source_name: The retired node being replaced, for naming connections in the report.
        incoming: Snapshotted incoming connections.
        released_outgoing: Outgoing connections whose downstream input slot has been freed.
            Slots that would not release are reported by the caller, which knows why.
        source: What the retired node can hand over, for its parameter renames.

    Returns:
        Descriptions of the connections that could not be rebuilt.
    """
    dropped: list[str] = []

    for upstream_node, upstream_param, source_param in incoming:
        new_param = source.parameter_renames.get(source_param, source_param)
        result = GriptapeNodes.handle_request(
            CreateConnectionRequest(
                source_node_name=upstream_node,
                source_parameter_name=upstream_param,
                target_node_name=new_name,
                target_parameter_name=new_param,
            )
        )
        if not isinstance(result, CreateConnectionResultSuccess):
            dropped.append(f"{upstream_node}.{upstream_param} -> {source_name}.{_readable_parameter(source_param)}")

    for source_param, downstream_node, downstream_param in released_outgoing:
        new_param = source.parameter_renames.get(source_param, source_param)
        result = GriptapeNodes.handle_request(
            CreateConnectionRequest(
                source_node_name=new_name,
                source_parameter_name=new_param,
                target_node_name=downstream_node,
                target_parameter_name=downstream_param,
            )
        )
        if not isinstance(result, CreateConnectionResultSuccess):
            dropped.append(f"{source_name}.{_readable_parameter(source_param)} -> {downstream_node}.{downstream_param}")

    return dropped
