import json
import re
from typing import Any

from base_analyze_media import BaseAnalyzeMedia


class IdentifyTimecodes(BaseAnalyzeMedia):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "Media Analysis/Google AI"
        self.description = "Identifies specific timecode markers in media content based on user prompts."

        # Update the existing prompt parameter with timecode-specific text
        prompt_param = self.get_parameter_by_name("prompt")
        if prompt_param:
            prompt_param.tooltip = (
                "What to look for in the media (e.g., 'man in a green hat', 'shot changes', 'dialog scenes')"
            )
            ui_options = prompt_param.ui_options
            ui_options["placeholder_text"] = "What should I look for in this media?"
            prompt_param.ui_options = ui_options

        # Since we're talking timecodes, we are only taking video or audio files
        media_param = self.get_parameter_by_name("media")
        if media_param:
            # Note: allowed_types might not be a settable attribute, this is just for documentation
            # The actual filtering should be handled by the UI or validation
            pass

        # The output is JSON, so let's modify the output parameter to be a JSON object
        output_param = self.get_parameter_by_name("output")
        if output_param:
            output_param.type = "json"
            output_param.tooltip = "The JSON output with timecode data"
            ui_options = output_param.ui_options
            ui_options["placeholder_text"] = "Timecode data will appear here"
            output_param.ui_options = ui_options

        # Let's hide the media_count and media_type parameters
        self.hide_parameter_by_name("media_count")
        self.hide_parameter_by_name("media_type")

    def _build_timecode_prompt(self, user_prompt: str) -> str:
        """Build a prompt that instructs Gemini to return timecode data in JSON format."""
        return f"""Identify each segment of time in the media where we can identify: {user_prompt}

First, analyze each video to determine its actual length, frame rate, and drop frame setting, then output the results as a JSON object with the following structure (return ONLY the JSON, no markdown formatting):

For a single video:
{{
  "title": "Video Title",
  "filename": "video.mp4",
  "time_format": "smpte",
  "length": <actual_length>,
  "rate": <actual_frame_rate>,
  "drop_frame": <actual_drop_frame_setting>,
  "chapters": [
    {{
      "id": "c001",
      "start": "00:00:00:00",
      "end": "00:00:12:12", 
      "title": "Brief descriptive title",
      "summary": "Detailed description of what happens in this segment",
      "tags": ["relevant", "tags"],
      "confidence": 0.86
    }}
  ]
}}

For multiple videos:
{{
  "videos": [
    {{
      "title": "Video 1 Title",
      "filename": "video1.mp4",
      "time_format": "smpte",
      "rate": <actual_frame_rate>,
      "drop_frame": <actual_drop_frame_setting>,
      "chapters": [
        {{
          "id": "c001",
          "start": "00:00:00:00",
          "end": "00:00:12:12", 
          "title": "Brief descriptive title",
          "summary": "Detailed description of what happens in this segment",
          "tags": ["relevant", "tags"],
          "confidence": 0.86
        }}
      ]
    }},
    {{
      "title": "Video 2 Title", 
      "filename": "video2.mp4",
      "time_format": "smpte",
      "rate": <actual_frame_rate>,
      "drop_frame": <actual_drop_frame_setting>,
      "chapters": [...]
    }}
  ]
}}

Guidelines:
- Analyze each video to determine its actual frame rate and drop frame setting, and use those values for the "rate" and "drop_frame" fields
- Use SMPTE timecode format (HH:MM:SS:FF) for start and end times based on the actual frame rate and drop frame setting
- For multiple videos, create a "videos" array with separate sections for each video
- Include descriptive titles for each video based on their content
- Include the actual filename for each video in the "filename" field
- Each chapter should represent a continuous segment where the specified content appears
- Include confidence scores (0.0-1.0) based on how certain you are about the identification
- Use descriptive titles and detailed summaries
- Add relevant tags to categorize each segment
- If no matching content is found in a video, return an empty chapters array for that video
- Ensure the JSON is valid and properly formatted
- Return ONLY the JSON object, no markdown code blocks or additional text"""

    def _analyze_multiple_media_with_gemini(
        self,
        client,
        all_media_sources: list,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Analyze media and extract timecode markers."""
        self._log(f"🤖 Analyzing {len(all_media_sources)} media item(s) for timecode markers")
        self._log("🔍 Starting timecode extraction...")

        # Build the specialized timecode prompt
        timecode_prompt = self._build_timecode_prompt(prompt)

        # Prepare the contents list
        contents = []

        # Add the timecode prompt
        contents.append(timecode_prompt)

        # Add each media source to contents
        for i, media_source in enumerate(all_media_sources):
            self._log(f"📁 Adding media item {i + 1}: {media_source['type']}")

            if media_source["type"] == "url":
                # Public URL - use directly
                contents.append({"file_data": {"file_uri": media_source["value"]}})
            else:  # GCS URI
                # GCS URI - use directly with MIME type
                contents.append(
                    {
                        "file_data": {
                            "file_uri": media_source["value"],
                            "mime_type": media_source.get("mime_type", "application/octet-stream"),
                        }
                    }
                )

        # Generate content with all media
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
            )

            if response.candidates and response.candidates[0].content:
                timecode_response = response.candidates[0].content.parts[0].text

                # Strip markdown code blocks if present
                cleaned_response = self._strip_markdown_json(timecode_response)

                # Try to parse the JSON response
                try:
                    parsed_json = json.loads(cleaned_response)
                    # Validate the structure - check for both single video and multiple video formats
                    if "chapters" in parsed_json and "time_format" in parsed_json:
                        # Single video format
                        self._log("✅ Successfully extracted timecode data (single video)")
                        return cleaned_response
                    if "videos" in parsed_json and isinstance(parsed_json["videos"], list):
                        # Multiple video format - validate each video has required fields
                        valid_videos = True
                        for i, video in enumerate(parsed_json["videos"]):
                            if "chapters" not in video or "time_format" not in video:
                                self._log(f"⚠️ Video {i + 1} missing required fields")
                                valid_videos = False
                                break
                        if valid_videos:
                            self._log("✅ Successfully extracted timecode data (multiple videos)")
                            return cleaned_response
                        self._log("⚠️ Some videos missing required fields, returning as-is")
                        return cleaned_response
                    self._log("⚠️ Response missing required fields, returning as-is")
                    return cleaned_response
                except json.JSONDecodeError:
                    self._log("⚠️ Response is not valid JSON, returning as-is")
                    return cleaned_response

            raise ValueError("No response generated from Gemini model")

        except Exception as e:
            # Handle "Service agents are being provisioned" error
            if "FAILED_PRECONDITION" in str(e) and "Service agents are being provisioned" in str(e):
                self._log("⚠️ Service agents are being provisioned. Retrying with inline data...")
                # For now, just re-raise the error since we don't have the original bytes
                raise
            # Re-raise other errors
            raise

    def _strip_markdown_json(self, text: str) -> str:
        """Remove markdown code blocks from JSON response."""
        # Remove ```json and ``` markers
        text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        # Also handle cases without language specification
        text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
        return text.strip()

    def process(self) -> Any:
        """Process the media and extract timecode markers."""
        # Just use the parent class process method - it will set the output parameter
        return super().process()
