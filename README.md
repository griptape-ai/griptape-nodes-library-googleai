# Griptape Nodes: Google AI Library

This library provides Griptape Nodes for interacting with Google AI services, including the powerful Veo video generation model and Imagen image generation models.

## Features

- **Veo Text-to-Video Generator**: Generate high-quality videos from text prompts using Google's state-of-the-art Veo model with advanced controls for aspect ratio, resolution, duration, and more.
- **Veo Image-to-Video Generator**: Transform images into videos using Google's Veo model with base64 encoding support.
- **Imagen Image Generator**: Generate images from text using Google's Imagen text-to-image models with comprehensive customization options.
- **Lyria Audio Generator**: Generate 30-second instrumental music using Google's Lyria model with creative prompt guidance to avoid copyright issues.
- **Describe Media**: Analyze images, videos, and audio using Google's Gemini model to answer questions about media content with support for multiple media types and conversation continuity via Google Cloud Storage.
- **Identify Timecodes**: Extract precise timecode markers from videos and audio using Gemini's analysis capabilities, outputting structured JSON with SMPTE timecodes, frame rates, and detailed chapter information.
- **Multi Video Display**: A dynamic node that displays video players in a grid layout with individual output ports for each video position.
- **Multi Audio Display**: A dynamic node that displays audio players in a grid layout with individual output ports for each audio position.

______________________________________________________________________

## 📦 Installation

### Prerequisites

- [Griptape Nodes](https://github.com/griptape-ai/griptape-nodes) installed and running
- Google Cloud account with billing enabled
- Google Cloud project with Vertex AI API enabled

### Install the Library

1. **Download the library files** to your Griptape Nodes workspace directory:
   ```bash
   # Navigate to your Griptape Nodes workspace directory
   cd `gtn config show workspace_directory`
   
   # Clone or download this library
   git clone https://github.com/your-username/griptape-nodes-library-googleai.git
   ```

2. **Add the library** in the Griptape Nodes Editor:
   * Open the Settings menu and navigate to the *Libraries* settings
   * Click on *+ Add Library* at the bottom of the settings panel
   * Enter the path to the library JSON file: **your Griptape Nodes Workspace directory**`/griptape-nodes-library-googleai/googleai/griptape_nodes_library.json`
   * You can check your workspace directory with `gtn config show workspace_directory`
   * Close the Settings Panel
   * Click on *Refresh Libraries*

3. **Verify installation** by checking that all Google AI nodes appear in your Griptape Nodes interface:
   - **Video/Google AI category**: Veo Text-To-Video, Veo Image-To-Video, Multi Video Display
   - **Image/Google AI category**: Imagen Image Generator
   - **Audio/Google AI category**: Lyria Audio Generator, Multi Audio Display
   - **Data/Google AI category**: Describe Media, Identify Timecodes

### Dependencies

This library automatically installs the following dependencies:
- `google-cloud-aiplatform`: For Vertex AI model access (Veo, Imagen, Lyria)
- `google-generativeai`: For Gemini model access (Describe Media, Identify Timecodes)
- `google-cloud-storage`: For optional media file storage and conversation continuity
- `requests`: For HTTP requests and file handling

______________________________________________________________________

## 🚀 Quick Start

Choose your path:

### Local Development (3 minutes)
**Testing on your machine?**

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="user:YOUR_EMAIL" --role="roles/aiplatform.user"
```

Set `GOOGLE_CLOUD_PROJECT_ID` in Griptape Nodes → Done! ✅

---

### Production Deployment

**Running in Google Cloud?**
- Cloud Run / GCE / GKE → Same as local setup, just attach a service account

**Running outside Google Cloud?**
- AWS / GitHub Actions / Azure → Use Workload Identity Federation (see below)

______________________________________________________________________

## 1. Authentication & Setup

This library supports multiple authentication methods. Choose the one that fits your needs:

| Method | Use Case | Setup Time |
|--------|----------|------------|
| **Local ADC** | Local testing | 3 min |
| **Cloud ADC** | Cloud Run/GCE/GKE | 5 min |
| **Workload Identity** | AWS/GitHub/Azure | 10 min |
| **Service Account** | Quick start | 5 min |

### Quick Setup for Common Scenarios

**Scenario 1: Testing Locally**
```bash
# One-time setup
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT_ID="your-project-id"
```

**Scenario 2: Deploying to Cloud Run**
```bash
# Create service account and deploy
gcloud iam service-accounts create my-sa
gcloud run deploy --service-account=my-sa@project.iam.gserviceaccount.com
```


______________________________________________________________________

## 2. Google Cloud Setup

### Step 1: Create a Service Account and Key

1. **Go to the Google Cloud Console**: Navigate to [IAM & Admin > Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts).
1. **Select your project** from the dropdown at the top of the page.
1. Click **+ CREATE SERVICE ACCOUNT**.
1. Give the service account a **Name** (e.g., `griptape-ai-generator`) and an optional description, then click **CREATE AND CONTINUE**.
1. **Grant access**: In the "Grant this service account access to project" step, click the **Role** field and search for and select the **Vertex AI User** role. This provides the necessary permissions to run AI Platform jobs. Click **CONTINUE**.
1. Click **DONE** to finish creating the service account.
1. **Create a key**: You will be returned to the list of service accounts. Find the one you just created and click on it.
1. Go to the **KEYS** tab.
1. Click **ADD KEY** and select **Create new key**.
1. Choose **JSON** as the key type and click **CREATE**. A JSON file will be downloaded to your computer. This is your credentials file.

### Step 2: Enable the Vertex AI API

1. **Return to the Google Cloud Console**
1. **Go to the API Library**: Navigate to [APIs & Services > Library](https://console.cloud.google.com/apis/library).
1. Select **+ Enable APIs and Services** at the top of the page.
1. Search for **Vertex AI API**.
1. Click on it and then click the **ENABLE** button if it is not already enabled for your project.

You are now ready to use the nodes!

______________________________________________________________________

## 3. Google Cloud Storage Setup (Optional - for Media Analysis)

The **Describe Media** and **Identify Timecodes** nodes can optionally use Google Cloud Storage (GCS) to store media files for conversation continuity. This allows you to reuse previously uploaded media without re-uploading.

### Step 1: Create a GCS Bucket

1. **Go to the Google Cloud Storage Console**: Navigate to [Cloud Storage > Create Bucket](https://console.cloud.google.com/storage/create-bucket)
1. **Create a new bucket** with the name `griptape-nodes`
1. **Choose your preferred region** (should match your Vertex AI region)
1. **Set access control** to "Uniform bucket-level access" (recommended)
1. **Click "Create"**

### Step 2: Grant Storage Permissions

1. **Go to your bucket's permissions**: In the Cloud Storage console, click on your `griptape-nodes` bucket
1. **Click "Permissions"** tab
1. **Click "Add"** to add a new principal
1. **Add your service account**: Enter your service account email (e.g., `griptape-ai-generator@griptape-nodes.iam.gserviceaccount.com`)
1. **Grant the "Storage Object Admin" role**: This provides the necessary permissions for uploading and managing objects
1. **Click "Save"**

### How It Works

- **Local Media**: When you choose local media files, they are automatically uploaded to GCS for future reuse
- **Public URLs**: Public URLs are used directly without GCS upload
- **Conversation Continuity**: Previously uploaded media can be referenced by GCS URI, avoiding re-upload
- **Fallback**: If GCS upload fails, the nodes automatically fall back to inline data transmission

**Note**: GCS is only used for local media files. Public URLs are passed directly to Gemini without GCS storage.

______________________________________________________________________

## 4. Example Video Generation Workflow

Here is an example of how to connect the nodes to generate and display videos:

1. **Add the `Veo Text-To-Video` node** to your workflow.
1. **Configure your prompt** and generation settings:
    - Write a creative text prompt
    - Choose the number of videos (1-4)
    - Select aspect ratio (16:9 or 9:16)
    - Set resolution (720p or 1080p)
    - Adjust duration (5-8 seconds)
    - Optionally add a negative prompt
    - Set a seed for reproducible results
1. **Add the `Display Video (Multi)` node**.
1. **Connect the `video_artifacts` output** from the `Veo Text-To-Video` to the `videos` input of the `Display Video (Multi)` node.
1. **Run the workflow!** The videos will appear in a grid layout with individual output ports for each video position.

![Example Veo Workflow](images/example_flow2.png)

### Advanced Features

- **Grid Display**: Videos are automatically arranged in a 2-column grid
- **Dynamic Outputs**: Individual output ports (`video_1_1`, `video_1_2`, etc.) are created for each generated video
- **Negative Prompts**: Specify what you don't want in your videos
- **Seed Control**: Use seeds for reproducible generation
- **Multiple Resolutions**: Support for 720p and 1080p output

## 5. Example Image-to-Video Workflow

1. **Add an image source** (e.g., `Imagen Image Generator` or import an image)
1. **Add the `Veo Image-To-Video` node**
1. **Connect the image** to the `image` input
1. **Configure settings**:
    - Add an optional animation prompt
    - Add a negative prompt if desired
    - Set aspect ratio and resolution
    - Configure seed for reproducibility
1. **Connect to `Display Video (Multi)`** for visualization
1. **Run the workflow!**

## 6. Example Image Generation Workflow

1. **Add the `Imagen Image Generator` node** to your workflow
1. **Configure your prompt** and generation settings:
    - Write a creative prompt
    - Choose from various Imagen models (3.0, 4.0 series)
    - Set aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4)
    - Configure advanced options like safety filtering and person generation
1. **Run the workflow!** The generated image will appear in the node in just a few seconds.

______________________________________________________________________

## 7. Nodes Reference

### Veo Text-To-Video Generator

**File**: `veo_video_generator.py`

The core video generation capability. Takes a text prompt and configuration options to generate up to 4 videos using Google's Veo model.

**Key Features**:

- Multiple Veo model options (3.0 series)
- Duration control (5-8 seconds)
- Resolution options (720p, 1080p)
- Aspect ratio support (16:9, 9:16)
- Negative prompts
- Seed-based reproducibility
- Grid output display

### Veo Image-To-Video Generator

**File**: `veo_image_to_video_generator.py`

Converts images into videos using Google's Veo model. Uses efficient base64 encoding for image input.

**Key Features**:

- Supports ImageArtifact and ImageUrlArtifact inputs
- Optional animation prompts
- Negative prompt support
- Seed control for reproducibility
- Same output format as text-to-video

### Imagen Image Generator

**File**: `imagen_image_generator.py`

Comprehensive image generation using Google's Imagen models with extensive customization options.

**Key Features**:

- Multiple Imagen models (3.0 and 4.0 series)
- Aspect ratio control
- Negative prompts (older models)
- Advanced safety and content filtering
- Person generation controls
- Prompt enhancement
- Seed-based reproducibility

### Lyria Audio Generator

**File**: `lyria_audio_generator.py`

Generate instrumental music using Google's Lyria model. Produces 30-second WAV audio clips at 48kHz with built-in copyright protection guidance.

**Key Features**:

- High-quality 48kHz WAV output
- Seed control for reproducibility
- Negative prompts for steering generation
- Intelligent recitation block detection with helpful suggestions
- Creative prompt guidance to avoid copyright issues

**Prompt Tips**:

- Use unique, specific descriptions instead of generic genres
- Combine unusual elements (e.g., "vintage synthesizer with rain sounds")
- Focus on textures and atmosphere rather than common musical terms
- Example: "ambient electronic soundscape with field recording elements"

### Multi Video Display

**File**: `multi_video_display.py`

A dynamic utility node for visualizing video outputs with intelligent grid layout and individual access ports.

**Key Features**:

- Automatic grid layout (2 columns)
- Dynamic output ports based on video count
- Real-time UI updates
- Individual video access (`video_1_1`, `video_1_2`, etc.)
- Debug status information

### Multi Audio Display

**File**: `multi_audio_display.py`

A dynamic utility node for visualizing audio outputs with intelligent grid layout and individual access ports.

**Key Features**:

- Automatic grid layout (2 columns)
- Dynamic output ports based on audio count
- Real-time UI updates
- Individual audio access (`audio_1_1`, `audio_1_2`, etc.)
- Compatible with AudioUrlArtifact format

______________________________________________________________________

## 7. Content Filtering

Google AI services include content filtering to prevent generation of harmful or inappropriate content. If your request is filtered:

- Check the logs output for filtering details and specific reasons
- Revise your prompt to avoid potentially harmful content
- Use negative prompts to steer away from problematic content
- Ensure your content complies with Google's usage policies

### Lyria Recitation Blocks

Lyria has additional copyright protection that may block prompts that seem too similar to existing copyrighted music. If you encounter recitation blocks:

**❌ Avoid Generic Terms**:

- "blues beat", "jazz song", "rock anthem"
- Simple genre names without unique elements

**✅ Use Creative Specificity**:

- "vintage synthesizer melodies with rain sounds and distant thunder"
- "acoustic guitar fingerpicking with subtle string arrangements"
- "ambient electronic soundscape with field recording elements"
- "minimalist piano with reverb over soft nature sounds"

**💡 Tips**:

- Combine unusual elements and textures
- Focus on atmosphere rather than genres
- Add environmental or unique sound elements
- Try the same prompt again - it sometimes works on retry!

______________________________________________________________________

## 8. Troubleshooting

### Common Issues

1. **Authentication errors**:

    - Verify service account has the Vertex AI User role
    - Check that JSON file path is correct and accessible
    - Ensure library settings are properly configured

1. **API not enabled**:

    - Make sure Vertex AI API is enabled for your project
    - Verify billing is set up for your Google Cloud project

1. **Content filtered**:

    - Revise prompts to avoid potentially harmful content
    - Use negative prompts to exclude problematic elements
    - Check logs for specific filtering reasons

1. **Model availability**:

    - Some models may be region-specific
    - Try different location settings if models aren't available

1. **Duration parameter errors**:

    - Note that image-to-video doesn't support duration control
    - Only text-to-video supports duration settings

### Performance Tips

- **Use appropriate resolutions**: 720p generates faster than 1080p
- **Batch multiple videos**: Generate up to 4 videos at once for efficiency
- **Use seeds**: For consistent results across multiple runs
- **Location selection**: Choose locations closer to your region for better performance

### Dependencies

The library automatically installs these required packages:

- `google-cloud-aiplatform`
- `google-generativeai`
- `google-cloud-storage`
- `requests`

**Note**: `opencv-python` has been removed to improve installation speed and reduce dependencies.

______________________________________________________________________

## 9. Library Settings Reference

Configure these settings in your Griptape Nodes library settings:

| Setting                               | Required   | Description                                            |
| ------------------------------------- | ---------- | ------------------------------------------------------ |
| `GOOGLE_WORKLOAD_IDENTITY_CONFIG_PATH`| Optional\* | Full path to workload identity federation config JSON  |
| `GOOGLE_SERVICE_ACCOUNT_FILE_PATH`    | Optional\* | Full path to service account JSON file                 |
| `GOOGLE_CLOUD_PROJECT_ID`             | Optional\* | Your Google Cloud project ID                           |
| `GOOGLE_APPLICATION_CREDENTIALS_JSON` | Optional\* | Complete JSON content of service account               |

\*At least one authentication method must be configured.

**Priority Order**: Workload identity federation → Service account file → JSON credentials → Application Default Credentials

______________________________________________________________________

## Version History

- **Latest**: Added library-level authentication, improved video display with dynamic outputs, enhanced parameter controls, removed opencv dependency
- **Previous**: Individual node authentication, basic video display, limited customization options
