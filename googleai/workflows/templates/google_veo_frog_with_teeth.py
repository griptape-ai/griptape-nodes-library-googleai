# /// script
# dependencies = []
#
# name = "google_veo_frog_with_teeth"
# schema_version = "0.13.0"
# engine_version_created_with = "0.63.0"
# node_libraries_referenced = [["Griptape Nodes Library", "0.50.0"], ["Google AI Library", "0.1.0"]]
# node_types_used = [["Google AI Library", "GeminiImageGenerator"], ["Google AI Library", "VeoTextToVideoWithRef"], ["Griptape Nodes Library", "Note"]]
# is_griptape_provided = true
# is_template = true
# image = "https://github.com/griptape-ai/griptape-nodes-library-googleai/blob/main/images/frog_with_teeth.webp?raw=true"
# description = "Demonstrates Text to Video (With Reference Images) node with Google's Veo model.""
# creation_date = 2025-11-11T01:35:50.850330Z
# last_modified_date = 2025-11-13T22:53:45.282999Z
#
# ///

import pickle

from griptape_nodes.node_library.library_registry import NodeMetadata
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest
from griptape_nodes.retained_mode.events.library_events import LoadLibrariesRequest
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.events.parameter_events import (
    SetParameterValueRequest,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

GriptapeNodes.handle_request(LoadLibrariesRequest())

context_manager = GriptapeNodes.ContextManager()

if not context_manager.has_current_workflow():
    context_manager.push_workflow(workflow_name="google_veo_frog_with_teeth")

"""
1. We've collated all of the unique parameter values into a dictionary so that we do not have to duplicate them.
   This minimizes the size of the code, especially for large objects like serialized image files.
2. We're using a prefix so that it's clear which Flow these values are associated with.
3. The values are serialized using pickle, which is a binary format. This makes them harder to read, but makes
   them consistently save and load. It allows us to serialize complex objects like custom classes, which otherwise
   would be difficult to serialize.
"""
top_level_unique_values_dict = {
    "b545fdac-c8b8-41ba-abb8-7860afc9e84c": pickle.loads(
        b"\x80\x04\x95\x13\x00\x00\x00\x00\x00\x00\x00\x8c\x0fLocal Execution\x94."
    ),
    "1d9beaec-ec3e-4795-a74c-6d37e5f90813": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x94."),
    "4f6a996a-a34c-4201-a4a2-2df79b0d3682": pickle.loads(
        b"\x80\x04\x95q\x00\x00\x00\x00\x00\x00\x00\x8cmThis workflow demonstrates the use of Reference Images for the Veo Text to Video (With Reference Images) node\x94."
    ),
    "541f3ffa-95a4-45bb-98eb-35483746b0e1": pickle.loads(
        b"\x80\x04\x95N\x00\x00\x00\x00\x00\x00\x00\x8cJGenerate a video of the frog that takes into account the reference images.\x94."
    ),
    "1b171537-4b50-44f2-baef-d5da7e2370ca": pickle.loads(
        b"\x80\x04\x95%\x00\x00\x00\x00\x00\x00\x00\x8c!Create the initial starting image\x94."
    ),
    "f7ff9c6b-37fc-4b82-8164-67928878c2e4": pickle.loads(
        b"\x80\x04\x959\x00\x00\x00\x00\x00\x00\x00\x8c5Create something specific about the frog - like teeth\x94."
    ),
    "9b696160-12a6-4aef-a82a-23be280139d4": pickle.loads(
        b"\x80\x04\x95\x1d\x02\x00\x00\x00\x00\x00\x00X\x16\x02\x00\x00Extreme close-up nighttime shot of a tiny green tree frog clinging to a translucent jungle leaf, illuminated by soft backlighting that highlights the intricate veins of the leaf. The frog's textured skin glistens with dew droplets, showcasing stunning detail and vibrant green tones. Dappled light filters through the dense, dark jungle environment, creating a dramatic and moody atmosphere. Foreground dust particles float gently, adding depth and realism to the scene. A composition worthy of an award-winning nature magazine cover.\x94."
    ),
    "90f10c8c-f1b1-4130-8046-156f7b01f30e": pickle.loads(b"\x80\x04]\x94."),
    "04194d5a-968d-44c8-80e5-301a2000b817": pickle.loads(
        b"\x80\x04\x95E\x00\x00\x00\x00\x00\x00\x00\x8cAGive the frog a giant smile, and give it human teeth with braces.\x94."
    ),
    "2a800d96-ba32-4d49-ba4e-8d913a8a9c9c": pickle.loads(b"\x80\x04]\x94."),
    "cab2ef82-f184-4c18-9432-a76076f0942a": pickle.loads(
        b"\x80\x04\x95K\x00\x00\x00\x00\x00\x00\x00\x8cGIt's raining, and the frog loves it. It looks at the camera and smiles.\x94."
    ),
}

"# Create the Flow, then do work within it as context."

flow0_name = GriptapeNodes.handle_request(
    CreateFlowRequest(parent_flow_name=None, set_as_new_context=False, metadata={})
).flow_name

with GriptapeNodes.ContextManager().flow(flow0_name):
    node0_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Note",
            specific_library_name="Griptape Nodes Library",
            node_name="Note",
            metadata={
                "position": {"x": 39.472653185919285, "y": 3946.158362657728},
                "tempId": "placing-1762972527319-2ddlk",
                "library_node_metadata": NodeMetadata(
                    category="misc",
                    description="Create a note node to provide helpful context in your workflow",
                    display_name="Note",
                    tags=None,
                    icon="notepad-text",
                    color=None,
                    group="create",
                    deprecation=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "Note",
                "showaddparameter": False,
                "size": {"width": 600, "height": 192},
                "category": "misc",
            },
            resolution="resolved",
            initial_setup=True,
        )
    ).node_name
    node1_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Note",
            specific_library_name="Griptape Nodes Library",
            node_name="Note_2",
            metadata={
                "position": {"x": 2669.320246403087, "y": 4166.746817177985},
                "tempId": "placing-1762975707225-1nqqnx",
                "library_node_metadata": NodeMetadata(
                    category="misc",
                    description="Create a note node to provide helpful context in your workflow",
                    display_name="Note",
                    tags=None,
                    icon="notepad-text",
                    color=None,
                    group="create",
                    deprecation=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "Note",
                "showaddparameter": False,
                "size": {"width": 1029, "height": 208},
                "category": "misc",
            },
            resolution="resolved",
            initial_setup=True,
        )
    ).node_name
    node2_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Note",
            specific_library_name="Griptape Nodes Library",
            node_name="Note_1",
            metadata={
                "position": {"x": 53.204941312303674, "y": 4229.5610110848465},
                "tempId": "placing-1762972527319-2ddlk",
                "library_node_metadata": NodeMetadata(
                    category="misc",
                    description="Create a note node to provide helpful context in your workflow",
                    display_name="Note",
                    tags=None,
                    icon="notepad-text",
                    color=None,
                    group="create",
                    deprecation=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "Note",
                "showaddparameter": False,
                "size": {"width": 600, "height": 192},
                "category": "misc",
            },
            resolution="resolved",
            initial_setup=True,
        )
    ).node_name
    node3_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Note",
            specific_library_name="Griptape Nodes Library",
            node_name="Note_3",
            metadata={
                "position": {"x": 975.0239695639402, "y": 5159.049021420599},
                "tempId": "placing-1762972527319-2ddlk",
                "library_node_metadata": NodeMetadata(
                    category="misc",
                    description="Create a note node to provide helpful context in your workflow",
                    display_name="Note",
                    tags=None,
                    icon="notepad-text",
                    color=None,
                    group="create",
                    deprecation=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "Note",
                "showaddparameter": False,
                "size": {"width": 600, "height": 192},
                "category": "misc",
            },
            resolution="resolved",
            initial_setup=True,
        )
    ).node_name
    node4_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="GeminiImageGenerator",
            specific_library_name="Google AI Library",
            node_name="Gemini Image Generator",
            metadata={
                "library_node_metadata": NodeMetadata(
                    category="image/googleai",
                    description="Generates images using Google's Gemini models.",
                    display_name="Gemini Image Generator",
                    tags=None,
                    icon=None,
                    color=None,
                    group=None,
                    deprecation=None,
                ),
                "library": "Google AI Library",
                "node_type": "GeminiImageGenerator",
                "position": {"x": 63.61828610454984, "y": 4449.382295809735},
                "size": {"width": 608, "height": 1080},
            },
            initial_setup=True,
        )
    ).node_name
    node5_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="GeminiImageGenerator",
            specific_library_name="Google AI Library",
            node_name="Gemini Image Generator_5",
            metadata={
                "library_node_metadata": NodeMetadata(
                    category="image/googleai",
                    description="Generates images using Google's Gemini models.",
                    display_name="Gemini Image Generator",
                    tags=None,
                    icon=None,
                    color=None,
                    group=None,
                    deprecation=None,
                ),
                "library": "Google AI Library",
                "node_type": "GeminiImageGenerator",
                "position": {"x": 959.5239695639402, "y": 5379.5610110848465},
                "size": {"width": 631, "height": 1095},
            },
            initial_setup=True,
        )
    ).node_name
    node6_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="VeoTextToVideoWithRef",
            specific_library_name="Google AI Library",
            node_name="Veo Text-To-Video (With Reference Images)",
            metadata={
                "library_node_metadata": NodeMetadata(
                    category="video/googleai",
                    description="Generates videos from text prompts with reference images using Google's Veo model. Only includes models that support reference images.",
                    display_name="Veo Text-To-Video (With Reference Images)",
                    tags=None,
                    icon=None,
                    color=None,
                    group=None,
                    deprecation=None,
                ),
                "library": "Google AI Library",
                "node_type": "VeoTextToVideoWithRef",
                "position": {"x": 2669.320246403087, "y": 4421.5610110848465},
                "size": {"width": 1008, "height": 2044},
            },
            initial_setup=True,
        )
    ).node_name
    """# Create the Flow, then do work within it as context."""
    flow1_name = GriptapeNodes.handle_request(
        CreateFlowRequest(parent_flow_name=None, set_as_new_context=False, metadata={})
    ).flow_name
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node4_name,
            source_parameter_name="image",
            target_node_name=node6_name,
            target_parameter_name="reference_image_1",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node5_name,
            source_parameter_name="image",
            target_node_name=node6_name,
            target_parameter_name="reference_image_2",
            initial_setup=True,
        )
    )
    with GriptapeNodes.ContextManager().node(node0_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="execution_environment",
                node_name=node0_name,
                value=top_level_unique_values_dict["b545fdac-c8b8-41ba-abb8-7860afc9e84c"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="job_group",
                node_name=node0_name,
                value=top_level_unique_values_dict["1d9beaec-ec3e-4795-a74c-6d37e5f90813"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node0_name,
                value=top_level_unique_values_dict["4f6a996a-a34c-4201-a4a2-2df79b0d3682"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node1_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="execution_environment",
                node_name=node1_name,
                value=top_level_unique_values_dict["b545fdac-c8b8-41ba-abb8-7860afc9e84c"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="job_group",
                node_name=node1_name,
                value=top_level_unique_values_dict["1d9beaec-ec3e-4795-a74c-6d37e5f90813"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node1_name,
                value=top_level_unique_values_dict["541f3ffa-95a4-45bb-98eb-35483746b0e1"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node2_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="execution_environment",
                node_name=node2_name,
                value=top_level_unique_values_dict["b545fdac-c8b8-41ba-abb8-7860afc9e84c"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="job_group",
                node_name=node2_name,
                value=top_level_unique_values_dict["1d9beaec-ec3e-4795-a74c-6d37e5f90813"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node2_name,
                value=top_level_unique_values_dict["1b171537-4b50-44f2-baef-d5da7e2370ca"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node3_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="execution_environment",
                node_name=node3_name,
                value=top_level_unique_values_dict["b545fdac-c8b8-41ba-abb8-7860afc9e84c"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="job_group",
                node_name=node3_name,
                value=top_level_unique_values_dict["1d9beaec-ec3e-4795-a74c-6d37e5f90813"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node3_name,
                value=top_level_unique_values_dict["f7ff9c6b-37fc-4b82-8164-67928878c2e4"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node4_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="prompt",
                node_name=node4_name,
                value=top_level_unique_values_dict["9b696160-12a6-4aef-a82a-23be280139d4"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="images",
                node_name=node4_name,
                value=top_level_unique_values_dict["90f10c8c-f1b1-4130-8046-156f7b01f30e"],
                initial_setup=True,
                is_output=True,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="logs",
                node_name=node4_name,
                value=top_level_unique_values_dict["1d9beaec-ec3e-4795-a74c-6d37e5f90813"],
                initial_setup=True,
                is_output=True,
            )
        )
    with GriptapeNodes.ContextManager().node(node5_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="prompt",
                node_name=node5_name,
                value=top_level_unique_values_dict["04194d5a-968d-44c8-80e5-301a2000b817"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="images",
                node_name=node5_name,
                value=top_level_unique_values_dict["2a800d96-ba32-4d49-ba4e-8d913a8a9c9c"],
                initial_setup=True,
                is_output=True,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="logs",
                node_name=node5_name,
                value=top_level_unique_values_dict["1d9beaec-ec3e-4795-a74c-6d37e5f90813"],
                initial_setup=True,
                is_output=True,
            )
        )
    with GriptapeNodes.ContextManager().node(node6_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="prompt",
                node_name=node6_name,
                value=top_level_unique_values_dict["cab2ef82-f184-4c18-9432-a76076f0942a"],
                initial_setup=True,
                is_output=False,
            )
        )
