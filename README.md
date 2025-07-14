# Griptape Nodes: Google AI Library

This library provides Griptape Nodes for interacting with Google AI services, including the powerful Veo video generation model and Imagen image generation models.

## Features

- **Veo Text-to-Video Generator**: Generate high-quality videos from text prompts using Google's state-of-the-art Veo model.
- **Veo Image-to-Video Generator**: Transform images into videos using Google's Veo model.
- **Imagen Image Generator**: Generate images from text using Google's Imagen text-to-image models.
- **Multi Video Display**: A dynamic node that displays video players for generated videos directly in the Griptape Nodes UI.
- **Last Frame Extractor**: Extract the last frame from a video and output it as an image for further processing.

---

## 1. Authentication

This library supports two methods for authenticating with Google Cloud.

### Method 1: Service Account File (Recommended for Servers)
This is the most explicit and secure method for production or automated workflows.

1.  **Create a Service Account and Key**: Follow the steps below to create a service account with the **Vertex AI User** role and download its JSON key file.
2.  **Provide the Path**: In the Google AI nodes, provide the full, absolute path to this JSON file in the `service_account_file` parameter. The node will use this specific identity to authenticate.

### Method 2: Application Default Credentials (Recommended for Local Development)
This method is convenient for local development and testing, as it uses the credentials from your local `gcloud` CLI.

1.  **Install the `gcloud` CLI**: If you haven't already, [install the Google Cloud CLI](https://cloud.google.com/sdk/docs/install).
2.  **Log in**: Run the following command in your terminal. This will open a browser window for you to log in with your Google account.
    ```bash
    gcloud auth login
    ```
3.  **Set your project**: Configure the `gcloud` CLI to use your target project.
    ```bash
    gcloud config set project YOUR_PROJECT_ID
    ```
4.  **Use the Node**:
    -   Leave the `service_account_file` parameter **empty**.
    -   Fill in the `project_id` parameter with your Google Cloud Project ID.
    -   The node will automatically detect and use your logged-in `gcloud` identity.

---

## 2. Setup & Configuration

### Step 1: Create a Service Account and Key (for Method 1)

1.  **Go to the Google Cloud Console**: Navigate to [IAM & Admin > Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts).
2.  **Select your project** from the dropdown at the top of the page.
3.  Click **+ CREATE SERVICE ACCOUNT**.
4.  Give the service account a **Name** (e.g., `griptape-ai-generator`) and an optional description, then click **CREATE AND CONTINUE**.
5.  **Grant access**: In the "Grant this service account access to project" step, click the **Role** field and search for and select the **Vertex AI User** role. This provides the necessary permissions to run AI Platform jobs. Click **CONTINUE**.
6.  Click **DONE** to finish creating the service account.
7.  **Create a key**: You will be returned to the list of service accounts. Find the one you just created and click on it.
8.  Go to the **KEYS** tab.
9.  Click **ADD KEY** and select **Create new key**.
10. Choose **JSON** as the key type and click **CREATE**. A JSON file will be downloaded to your computer. This is your credentials file.

### Step 2: Enable the Vertex AI API

1. **Return to the Google Cloud Console**
2.  **Go to the API Library**: Navigate to [APIs & Services > Library](https://console.cloud.google.com/apis/library).
3.  Select **+ Enable APIs and Services** at the top of the page.
4.  Search for **Vertex AI API**.
5.  Click on it and then click the **ENABLE** button if it is not already enabled for your project.

You are now ready to use the nodes!

---

## 3. Example Video Generation Workflow

Here is an example of how to connect the nodes to generate and display videos.

1.  Add the `Veo Text-To-Video` node to your workflow.
2.  Expand the `GOOGLECONFIG` section at the top of the node and **Choose your authentication method**:
    -   Either provide the path to your `service_account_file` (leaving `project_id` blank).
    -   Or leave `service_account_file` blank and provide your `project_id`.
3.  Write a creative prompt.
4.  Choose the number of videos to generate.
5.  Add the `Display Video (Multi)` node.
6.  Connect the `video_artifacts` output from the `Veo Text-To-Video` to the `videos` input of the `Display Video (Multi)` node.
7.  Run the workflow! The `Display Video (Multi)` node will display video players for each generated video.

![Example Veo Workflow](images/example_flow2.png)

## 4. Example Image Generation Workflow

1. Add the `Imagen Image Generator` node to your workflow
2.  Expand the `GOOGLECONFIG` section at the top of the node and **Choose your authentication method**:
    -   Either provide the path to your `service_account_file` (leaving `project_id` blank).
    -   Or leave `service_account_file` blank and provide your `project_id`.
3.  Write a creative prompt.
4.  Optionally, select the model that you wish to use and set other configuration options.
5.  Run the workflow! The generated image will appear in the node in just a few seconds.

---

## 5. Nodes

### Veo Text-To-Video
This node is the core video generation capability. It takes a service account file, a text prompt, and other configuration options to generate one or more videos using the Google Veo model via the Vertex AI API. It outputs a list of video artifacts.

**File**: `veo_video_generator.py`

### Veo Image-To-Video
Converts images into videos using Google's Veo model. Takes an image input along with optional prompts and configuration to generate videos from the provided image.

**File**: `veo_image_to_video_generator.py`

### Imagen Image Generator
Provides support for image generation with Google's Imagen family of text-to-image models. It takes a service account file, a text prompt, and other configuration options to generate an image using a Google Imagen model of your choice via the Vertex AI API. It outputs an image.

**File**: `imagen_image_generator.py`

### Display Video (Multi)
A utility node designed to visualize the output of the video generators. It accepts video artifacts (single or list) and displays interactive video players directly in the node's UI, with output ports to pass videos to downstream nodes.

**File**: `multi_video_display.py`

### Last Frame Extractor
Extracts the last frame from a video and outputs it as an image. Useful for creating thumbnails or extracting final frames for further image processing workflows.

**File**: `last_frame.py`

## 6. Content Filtering

Google AI services include content filtering to prevent generation of harmful or inappropriate content. If your request is filtered:

- Check the logs output for filtering details
- Revise your prompt to avoid potentially harmful content
- Ensure your content complies with Google's usage policies

## 7. Troubleshooting

### Common Issues

1. **Authentication errors**: Ensure your service account has the Vertex AI User role
2. **API not enabled**: Make sure Vertex AI API is enabled for your project
3. **Content filtered**: Revise prompts to avoid potentially harmful content
4. **File not found**: Check that service account JSON file path is correct

### Dependencies

The library requires these Python packages:
- `google-cloud-aiplatform`
- `google-generativeai`
- `google-cloud-storage`
- `requests`
- `opencv-python`

These are automatically installed when you add the library to your project.