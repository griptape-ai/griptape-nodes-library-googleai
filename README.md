# Griptape Nodes: Google AI Library

This library provides Griptape Nodes for interacting with Google AI services, including the powerful Veo video generation model.

## Features

- **Veo Video Generator**: Generate high-quality videos from text prompts using Google's state-of-the-art Veo model.
- **Video Display**: A dynamic node that displays video players for generated videos directly in the Griptape Nodes UI.

---

## 1. Setup & Configuration

To use this library, you first need to configure a Google Cloud project with the necessary permissions and credentials.

### Step 1: Create a Service Account and Key

1.  **Go to the Google Cloud Console**: Navigate to [IAM & Admin > Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts).
2.  **Select your project** from the dropdown at the top of the page.
3.  Click **+ CREATE SERVICE ACCOUNT**.
4.  Give the service account a **Name** (e.g., `griptape-veo-generator`) and an optional description, then click **CREATE AND CONTINUE**.
5.  **Grant access**: In the "Grant this service account access to project" step, click the **Role** field and search for and select the **Vertex AI User** role. This provides the necessary permissions to run AI Platform jobs. Click **CONTINUE**.
6.  Click **DONE** to finish creating the service account.
7.  **Create a key**: You will be returned to the list of service accounts. Find the one you just created and click on it.
8.  Go to the **KEYS** tab.
9.  Click **ADD KEY** and select **Create new key**.
10. Choose **JSON** as the key type and click **CREATE**. A JSON file will be downloaded to your computer. This is your credentials file.

### Step 2: Enable the Vertex AI API

1.  **Go to the API Library**: Navigate to [APIs & Services > Library](https://console.cloud.google.com/apis/library).
2.  Search for **Vertex AI API**.
3.  Click on it and then click the **ENABLE** button if it is not already enabled for your project.

You are now ready to use the nodes!

---

## 2. Example Workflow

Here is an example of how to connect the nodes to generate and display videos.

1.  Add the `Veo Video Generator` node to your workflow.
2.  In the `service_account_file` parameter, provide the **full path** to the JSON key file you downloaded in the setup steps.
3.  Write a creative prompt.
4.  Choose the number of videos to generate.
5.  Add the `Video Display` node.
6.  Connect the `video_artifacts` output from the `Veo Video Generator` to the `video_artifacts` input of the `Video Display` node.
7.  Run the workflow! The `Video Display` node will dynamically create a video player for each generated video.

![Example Veo Workflow](images/example_flow2.png)

---

## 3. Nodes

### Veo Video Generator
This node is the core of the library. It takes a service account file, a text prompt, and other configuration options to generate one or more videos using the Google Veo model via the Vertex AI API. It outputs a list of video artifacts.

### Video Display
A utility node designed to visualize the output of the video generator. It accepts a list of video artifacts and dynamically creates an interactive video player for each one directly in the node's UI, complete with output ports to pass individual videos to downstream nodes.
