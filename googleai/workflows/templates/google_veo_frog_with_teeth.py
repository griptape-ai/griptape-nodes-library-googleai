# /// script
# dependencies = []
#
# [tool.griptape-nodes]
# name = "google_veo_frog_with_teeth"
# schema_version = "0.13.0"
# engine_version_created_with = "0.63.0"
# node_libraries_referenced = [["Griptape Nodes Library", "0.50.0"], ["Google AI Library", "0.1.0"]]
# node_types_used = [["Google AI Library", "GeminiImageGenerator"], ["Google AI Library", "VeoTextToVideoWithRef"], ["Griptape Nodes Library", "Note"]]
# is_griptape_provided = true
# is_template = true
# description = Demonstrates Text to Video (With Reference Images) node with Google's Veo model.
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
    AddParameterToNodeRequest,
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
    "66080bcb-aa93-4277-b4b2-0ef542ae3a7a": pickle.loads(
        b"\x80\x04\x95\x13\x00\x00\x00\x00\x00\x00\x00\x8c\x0fLocal Execution\x94."
    ),
    "19c3eb34-b657-4be9-bee3-f6211ac450d6": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x94."),
    "1fc2a1a4-523e-444b-b72e-b37b198688ec": pickle.loads(
        b"\x80\x04\x95q\x00\x00\x00\x00\x00\x00\x00\x8cmThis workflow demonstrates the use of Reference Images for the Veo Text to Video (With Reference Images) node\x94."
    ),
    "a894f949-33a9-4847-85c6-2cdbf8cfbe3d": pickle.loads(
        b"\x80\x04\x95N\x00\x00\x00\x00\x00\x00\x00\x8cJGenerate a video of the frog that takes into account the reference images.\x94."
    ),
    "8b636c33-014b-4c24-9aeb-05528f22ca39": pickle.loads(
        b"\x80\x04\x95%\x00\x00\x00\x00\x00\x00\x00\x8c!Create the initial starting image\x94."
    ),
    "20482eea-cbb5-49c7-b6f7-90b8fd568af6": pickle.loads(
        b"\x80\x04\x959\x00\x00\x00\x00\x00\x00\x00\x8c5Create something specific about the frog - like teeth\x94."
    ),
    "fceb8226-ca6d-44f0-8781-0a40d3e4a8f4": pickle.loads(
        b"\x80\x04\x95l\x00\x00\x00\x00\x00\x00\x00\x8chFrog wistfully talking about its typical day, as if it's being interviewed.  Keep the same camera angle.\x94."
    ),
    "b45a6faf-c514-4813-98b0-bae0d3bc41cd": pickle.loads(
        b"\x80\x04\x95C\x00\x00\x00\x00\x00\x00\x00\x8c?Give the frog a bit smile, but give it human teeth with braces.\x94."
    ),
    "ce24b1c4-5816-495a-bdcc-c73eb06ee021": pickle.loads(b"\x80\x04]\x94."),
    "fb9c1b3e-4faf-4f40-a015-67b3d6c9ca1e": pickle.loads(b"\x80\x04]\x94."),
    "affa0130-d647-43d6-aea3-500a25bdd5a1": pickle.loads(
        b"\x80\x04\x95\x1d\x02\x00\x00\x00\x00\x00\x00X\x16\x02\x00\x00Extreme close-up nighttime shot of a tiny green tree frog clinging to a translucent jungle leaf, illuminated by soft backlighting that highlights the intricate veins of the leaf. The frog's textured skin glistens with dew droplets, showcasing stunning detail and vibrant green tones. Dappled light filters through the dense, dark jungle environment, creating a dramatic and moody atmosphere. Foreground dust particles float gently, adding depth and realism to the scene. A composition worthy of an award-winning nature magazine cover.\x94."
    ),
    "000b006c-11e0-4d60-863e-eca4f172a3e5": pickle.loads(b"\x80\x04]\x94."),
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
            node_type="VeoTextToVideoWithRef",
            specific_library_name="Google AI Library",
            node_name="Veo Text-To-Video (With Reference Images)",
            metadata={
                "library_node_metadata": {
                    "category": "video/googleai",
                    "description": "Generates videos from text prompts with reference images using Google's Veo model. Only includes models that support reference images.",
                },
                "library": "Google AI Library",
                "node_type": "VeoTextToVideoWithRef",
                "position": {"x": 2669.320246403087, "y": 4421.5610110848465},
                "size": {"width": 996, "height": 1662},
                "showaddparameter": False,
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
    with GriptapeNodes.ContextManager().node(node5_name):
        GriptapeNodes.handle_request(
            AddParameterToNodeRequest(
                parameter_name="input_images_ParameterListUniqueParamID_9e60903a8305492fad0a5c97cd65f5be",
                tooltip="Up to 3 input images (png/jpeg/webp, ≤ 7 MB each). These visual references are used by the model to guide image generation, similar to image-to-image generation.",
                type="ImageArtifact",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                output_type="ImageArtifact",
                ui_options={},
                mode_allowed_input=True,
                mode_allowed_property=True,
                mode_allowed_output=False,
                is_user_defined=True,
                settable=True,
                parent_container_name="input_images",
                initial_setup=True,
            )
        )
    node6_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="GeminiImageGenerator",
            specific_library_name="Google AI Library",
            node_name="Gemini Image Generator",
            metadata={
                "library_node_metadata": {
                    "category": "image/googleai",
                    "description": "Generates images using Google's Gemini models.",
                },
                "library": "Google AI Library",
                "node_type": "GeminiImageGenerator",
                "position": {"x": 63.61828610454984, "y": 4449.382295809735},
                "size": {"width": 608, "height": 1080},
                "showaddparameter": False,
                "category": "image/googleai",
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
            source_node_name=node6_name,
            source_parameter_name="image",
            target_node_name=node4_name,
            target_parameter_name="reference_image_1",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node5_name,
            source_parameter_name="image",
            target_node_name=node4_name,
            target_parameter_name="reference_image_2",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node6_name,
            source_parameter_name="image",
            target_node_name=node5_name,
            target_parameter_name="input_images_ParameterListUniqueParamID_9e60903a8305492fad0a5c97cd65f5be",
            initial_setup=True,
        )
    )
    with GriptapeNodes.ContextManager().node(node0_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="execution_environment",
                node_name=node0_name,
                value=top_level_unique_values_dict["66080bcb-aa93-4277-b4b2-0ef542ae3a7a"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="job_group",
                node_name=node0_name,
                value=top_level_unique_values_dict["19c3eb34-b657-4be9-bee3-f6211ac450d6"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node0_name,
                value=top_level_unique_values_dict["1fc2a1a4-523e-444b-b72e-b37b198688ec"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node1_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="execution_environment",
                node_name=node1_name,
                value=top_level_unique_values_dict["66080bcb-aa93-4277-b4b2-0ef542ae3a7a"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="job_group",
                node_name=node1_name,
                value=top_level_unique_values_dict["19c3eb34-b657-4be9-bee3-f6211ac450d6"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node1_name,
                value=top_level_unique_values_dict["a894f949-33a9-4847-85c6-2cdbf8cfbe3d"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node2_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="execution_environment",
                node_name=node2_name,
                value=top_level_unique_values_dict["66080bcb-aa93-4277-b4b2-0ef542ae3a7a"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="job_group",
                node_name=node2_name,
                value=top_level_unique_values_dict["19c3eb34-b657-4be9-bee3-f6211ac450d6"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node2_name,
                value=top_level_unique_values_dict["8b636c33-014b-4c24-9aeb-05528f22ca39"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node3_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="execution_environment",
                node_name=node3_name,
                value=top_level_unique_values_dict["66080bcb-aa93-4277-b4b2-0ef542ae3a7a"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="job_group",
                node_name=node3_name,
                value=top_level_unique_values_dict["19c3eb34-b657-4be9-bee3-f6211ac450d6"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node3_name,
                value=top_level_unique_values_dict["20482eea-cbb5-49c7-b6f7-90b8fd568af6"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node4_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="prompt",
                node_name=node4_name,
                value=top_level_unique_values_dict["fceb8226-ca6d-44f0-8781-0a40d3e4a8f4"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node5_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="prompt",
                node_name=node5_name,
                value=top_level_unique_values_dict["b45a6faf-c514-4813-98b0-bae0d3bc41cd"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_images",
                node_name=node5_name,
                value=top_level_unique_values_dict["ce24b1c4-5816-495a-bdcc-c73eb06ee021"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="images",
                node_name=node5_name,
                value=top_level_unique_values_dict["fb9c1b3e-4faf-4f40-a015-67b3d6c9ca1e"],
                initial_setup=True,
                is_output=True,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="logs",
                node_name=node5_name,
                value=top_level_unique_values_dict["19c3eb34-b657-4be9-bee3-f6211ac450d6"],
                initial_setup=True,
                is_output=True,
            )
        )
    with GriptapeNodes.ContextManager().node(node6_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="prompt",
                node_name=node6_name,
                value=top_level_unique_values_dict["affa0130-d647-43d6-aea3-500a25bdd5a1"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="images",
                node_name=node6_name,
                value=top_level_unique_values_dict["000b006c-11e0-4d60-863e-eca4f172a3e5"],
                initial_setup=True,
                is_output=True,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="logs",
                node_name=node6_name,
                value=top_level_unique_values_dict["19c3eb34-b657-4be9-bee3-f6211ac450d6"],
                initial_setup=True,
                is_output=True,
            )
        )
