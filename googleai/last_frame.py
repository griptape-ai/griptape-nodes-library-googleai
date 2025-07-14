import io
import time
import requests
from typing import Any
from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

# Attempt to import video processing library
try:
    import cv2
    import numpy as np
    CV2_INSTALLED = True
except ImportError:
    CV2_INSTALLED = False


class VideoUrlArtifact:
    """
    Artifact that contains a URL to a video.
    """
    def __init__(self, value: str, name: str | None = None):
        self.value = value
        self.name = name or self.__class__.__name__


class LastFrameExtractor(DataNode):
    """
    A node that extracts the last frame from a video and outputs it as an image.
    """
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "Google AI"
        self.description = "Extracts the last frame from a video and outputs it as an image."

        # Input parameter for video
        self.add_parameter(
            Parameter(
                name="video_input",
                input_types=["VideoUrlArtifact"],
                type="VideoUrlArtifact",
                tooltip="The video to extract the last frame from.",
                allowed_modes={ParameterMode.INPUT},
            )
        )

        # Output parameter for the extracted frame
        self.add_parameter(
            Parameter(
                name="last_frame_image",
                output_type="ImageUrlArtifact",
                type="ImageUrlArtifact",
                tooltip="The last frame of the video as an image.",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"is_full_width": True, "pulse_on_run": True},
            )
        )

        # Status output for debugging
        self.add_parameter(
            Parameter(
                name="status",
                output_type="str",
                type="str",
                tooltip="Status messages from the frame extraction process.",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"multiline": True},
            )
        )

    def _log(self, message: str):
        """Append a message to the status output parameter."""
        print(message)
        self.append_value_to_parameter("status", message + "\n")

    def _download_video(self, video_url: str) -> bytes:
        """Download video from URL and return bytes."""
        self._log(f"📥 Downloading video from: {video_url}")
        
        try:
            response = requests.get(video_url, timeout=30)
            response.raise_for_status()
            self._log(f"✅ Video downloaded successfully ({len(response.content)} bytes)")
            return response.content
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download video: {e}")

    def _extract_last_frame(self, video_bytes: bytes) -> bytes:
        """Extract the last frame from video bytes and return as image bytes."""
        if not CV2_INSTALLED:
            raise RuntimeError("OpenCV (cv2) is required for video processing. Please install opencv-python.")

        self._log("🎬 Extracting last frame from video...")
        
        # Write video bytes to a temporary buffer
        video_buffer = io.BytesIO(video_bytes)
        
        # Create a temporary file-like object that OpenCV can read
        temp_video_path = f"/tmp/temp_video_{int(time.time())}.mp4"
        
        try:
            # Write bytes to temporary file
            with open(temp_video_path, 'wb') as f:
                f.write(video_bytes)
            
            # Open video with OpenCV
            cap = cv2.VideoCapture(temp_video_path)
            
            if not cap.isOpened():
                raise RuntimeError("Could not open video file")
            
            # Get total frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            self._log(f"📊 Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
            
            if total_frames == 0:
                raise RuntimeError("Video has no frames")
            
            # Seek to the last frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            
            # Read the last frame
            ret, frame = cap.read()
            
            if not ret or frame is None:
                raise RuntimeError("Could not read the last frame")
            
            self._log(f"✅ Last frame extracted (shape: {frame.shape})")
            
            # Encode frame as PNG
            success, buffer = cv2.imencode('.png', frame)
            
            if not success:
                raise RuntimeError("Failed to encode frame as PNG")
            
            cap.release()
            
            # Clean up temporary file
            try:
                import os
                os.remove(temp_video_path)
            except:
                pass  # Don't fail if cleanup fails
            
            return buffer.tobytes()
            
        except Exception as e:
            # Clean up on error
            try:
                import os
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
            except:
                pass
            raise e

    def process(self) -> None:
        try:
            # Get input video
            video_input = self.get_parameter_value("video_input")
            
            if not video_input:
                self._log("❌ ERROR: No video input provided.")
                return
            
            # Handle different input types
            video_url = None
            if isinstance(video_input, VideoUrlArtifact):
                video_url = video_input.value
            elif hasattr(video_input, 'value'):
                video_url = video_input.value
            elif isinstance(video_input, str):
                video_url = video_input
            else:
                self._log(f"❌ ERROR: Unsupported video input type: {type(video_input)}")
                return
            
            if not video_url:
                self._log("❌ ERROR: No video URL found in input.")
                return
            
            self._log(f"🎯 Processing video: {video_url}")
            
            # Download video
            video_bytes = self._download_video(video_url)
            
            # Extract last frame
            frame_bytes = self._extract_last_frame(video_bytes)
            
            # Save frame to static storage
            timestamp = int(time.time())
            filename = f"last_frame_{timestamp}.png"
            self._log(f"💾 Saving frame as {filename}...")
            
            static_files_manager = GriptapeNodes.StaticFilesManager()
            image_url = static_files_manager.save_static_file(frame_bytes, filename)
            
            # Create output artifact
            image_artifact = ImageUrlArtifact(value=image_url, name=filename)
            self.parameter_output_values["last_frame_image"] = image_artifact
            
            self._log(f"🎉 SUCCESS! Last frame extracted and saved: {image_url}")
            
        except Exception as e:
            error_msg = f"❌ ERROR: {str(e)}"
            self._log(error_msg)
            import traceback
            self._log(f"Traceback: {traceback.format_exc()}") 