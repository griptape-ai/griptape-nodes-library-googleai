# Griptape Nodes: Google AI Library

This library provides Griptape Nodes for interacting with Google AI services, including the powerful Veo video generation model and Imagen image generation models.

## Features

- **Veo Text-to-Video Generator**: Generate high-quality videos from text prompts using Google's state-of-the-art Veo model with advanced controls for aspect ratio, resolution, duration, and more.
- **Veo Image-to-Video Generator**: Transform images into videos using Google's Veo model with base64 encoding support.
- **Imagen Image Generator**: Generate images from text using Google's Imagen text-to-image models with comprehensive customization options.
- **Lyria Audio Generator**: Generate 30-second instrumental music using Google's Lyria model with creative prompt guidance to avoid copyright issues.
- **Multi Video Display**: A dynamic node that displays video players in a grid layout with individual output ports for each video position.
- **Multi Audio Display**: A dynamic node that displays audio players in a grid layout with individual output ports for each audio position.

---

## 1. Authentication & Setup

This library uses **library-level settings** for authentication, making it easy to configure once and use across all Google AI nodes.

### Step 1: Configure Library Settings

When you install this library in Griptape Nodes, you'll need to configure your Google Cloud credentials in the library settings:

1. **Open Griptape Nodes Settings**
2. **Navigate to Libraries** 
3. **Find "Google AI" library settings**
4. **Configure one of the authentication methods below**

### Method 1: Service Account File (Recommended)

1. **Create a Service Account** (see detailed steps below)
2. **Download the JSON key file**
3. **Set the file path in library settings**:
   - `GOOGLE_SERVICE_ACCOUNT_FILE_PATH`: Full path to your service account JSON file

### Method 2: JSON Credentials

1. **Create a Service Account** (see detailed steps below)
2. **Copy the entire JSON content**
3. **Set the credentials in library settings**:
   - `GOOGLE_CLOUD_PROJECT_ID`: Your Google Cloud project ID
   - `GOOGLE_APPLICATION_CREDENTIALS_JSON`: Paste the entire JSON content

### Method 3: Application Default Credentials (Local Development)

1. **Install and configure gcloud CLI**:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```
2. **Set the project ID in library settings**:
   - `GOOGLE_CLOUD_PROJECT_ID`: Your Google Cloud project ID

---

## 2. Google Cloud Setup

### Step 1: Create a Service Account and Key

1. **Go to the Google Cloud Console**: Navigate to [IAM & Admin > Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts).
2. **Select your project** from the dropdown at the top of the page.
3. Click **+ CREATE SERVICE ACCOUNT**.
4. Give the service account a **Name** (e.g., `griptape-ai-generator`) and an optional description, then click **CREATE AND CONTINUE**.
5. **Grant access**: In the "Grant this service account access to project" step, click the **Role** field and search for and select the **Vertex AI User** role. This provides the necessary permissions to run AI Platform jobs. Click **CONTINUE**.
6. Click **DONE** to finish creating the service account.
7. **Create a key**: You will be returned to the list of service accounts. Find the one you just created and click on it.
8. Go to the **KEYS** tab.
9. Click **ADD KEY** and select **Create new key**.
10. Choose **JSON** as the key type and click **CREATE**. A JSON file will be downloaded to your computer. This is your credentials file.

### Step 2: Enable the Vertex AI API

1. **Return to the Google Cloud Console**
2. **Go to the API Library**: Navigate to [APIs & Services > Library](https://console.cloud.google.com/apis/library).
3. Select **+ Enable APIs and Services** at the top of the page.
4. Search for **Vertex AI API**.
5. Click on it and then click the **ENABLE** button if it is not already enabled for your project.

You are now ready to use the nodes!

---

## 3. Example Video Generation Workflow

Here is an example of how to connect the nodes to generate and display videos:

1. **Add the `Veo Text-To-Video` node** to your workflow.
2. **Configure your prompt** and generation settings:
   - Write a creative text prompt
   - Choose the number of videos (1-4)
   - Select aspect ratio (16:9 or 9:16)
   - Set resolution (720p or 1080p)
   - Adjust duration (5-8 seconds)
   - Optionally add a negative prompt
   - Set a seed for reproducible results
3. **Add the `Display Video (Multi)` node**.
4. **Connect the `video_artifacts` output** from the `Veo Text-To-Video` to the `videos` input of the `Display Video (Multi)` node.
5. **Run the workflow!** The videos will appear in a grid layout with individual output ports for each video position.

![Example Veo Workflow](images/example_flow2.png)

### Advanced Features

- **Grid Display**: Videos are automatically arranged in a 2-column grid
- **Dynamic Outputs**: Individual output ports (`video_1_1`, `video_1_2`, etc.) are created for each generated video
- **Negative Prompts**: Specify what you don't want in your videos
- **Seed Control**: Use seeds for reproducible generation
- **Multiple Resolutions**: Support for 720p and 1080p output

## 4. Example Image-to-Video Workflow

1. **Add an image source** (e.g., `Imagen Image Generator` or import an image)
2. **Add the `Veo Image-To-Video` node**
3. **Connect the image** to the `image` input
4. **Configure settings**:
   - Add an optional animation prompt
   - Add a negative prompt if desired
   - Set aspect ratio and resolution
   - Configure seed for reproducibility
5. **Connect to `Display Video (Multi)`** for visualization
6. **Run the workflow!**

## 5. Example Image Generation Workflow

1. **Add the `Imagen Image Generator` node** to your workflow
2. **Configure your prompt** and generation settings:
   - Write a creative prompt
   - Choose from various Imagen models (3.0, 4.0 series)
   - Set aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4)
   - Configure advanced options like safety filtering and person generation
3. **Run the workflow!** The generated image will appear in the node in just a few seconds.

---

## 6. Nodes Reference

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

---

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

---

## 8. Troubleshooting

### Common Issues

1. **Authentication errors**: 
   - Verify service account has the Vertex AI User role
   - Check that JSON file path is correct and accessible
   - Ensure library settings are properly configured

2. **API not enabled**: 
   - Make sure Vertex AI API is enabled for your project
   - Verify billing is set up for your Google Cloud project

3. **Content filtered**: 
   - Revise prompts to avoid potentially harmful content
   - Use negative prompts to exclude problematic elements
   - Check logs for specific filtering reasons

4. **Model availability**: 
   - Some models may be region-specific
   - Try different location settings if models aren't available

5. **Duration parameter errors**: 
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

---

## 9. Library Settings Reference

Configure these settings in your Griptape Nodes library settings:

| Setting | Required | Description |
|---------|----------|-------------|
| `GOOGLE_SERVICE_ACCOUNT_FILE_PATH` | Optional* | Full path to service account JSON file |
| `GOOGLE_CLOUD_PROJECT_ID` | Optional* | Your Google Cloud project ID |
| `GOOGLE_APPLICATION_CREDENTIALS_JSON` | Optional* | Complete JSON content of service account |

*At least one authentication method must be configured.

**Priority Order**: Service account file → JSON credentials → Application Default Credentials

---

## Version History

- **Latest**: Added library-level authentication, improved video display with dynamic outputs, enhanced parameter controls, removed opencv dependency
- **Previous**: Individual node authentication, basic video display, limited customization options