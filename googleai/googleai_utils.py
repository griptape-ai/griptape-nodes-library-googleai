"""Common utilities for GoogleAI nodes."""

import json
import os
from collections.abc import Callable
from typing import Any


def detect_image_mime_from_bytes(data: bytes) -> str | None:
    """Detect image MIME type from magic bytes.

    Returns the detected MIME type or None if not recognized.
    """
    if len(data) < 12:
        return None
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:2] == b"\xff\xd8":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1"):
        return "image/heic"
    if data[4:12] in (b"ftypheif", b"ftypmif1"):
        return "image/heif"
    return None


try:
    import io as _io

    from PIL import Image as PILImage

    PIL_INSTALLED = True
except Exception:
    PIL_INSTALLED = False

try:
    import google.auth
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    GOOGLE_AUTH_INSTALLED = True
except ImportError:
    GOOGLE_AUTH_INSTALLED = False


class GoogleAuthHelper:
    """Helper class for Google Cloud authentication with support for multiple auth methods.

    Supports (in priority order):
    1. Workload Identity Federation - Cross-cloud environments (AWS, GitHub Actions, Azure)
       Requires: GOOGLE_WORKLOAD_IDENTITY_CONFIG_PATH

    2. Service Account File - Explicit key file authentication
       Requires: GOOGLE_SERVICE_ACCOUNT_FILE_PATH

    3. Service Account JSON - Inline JSON credentials
       Requires: GOOGLE_APPLICATION_CREDENTIALS_JSON

    4. Application Default Credentials (ADC) - Auto-detected from environment
       Requires: GOOGLE_CLOUD_PROJECT_ID
       Works with:
       - Cloud Run, GKE, GCE (metadata server)
       - Local development (gcloud auth application-default login)
       - GOOGLE_APPLICATION_CREDENTIALS environment variable
       - Any environment with configured ADC
    """

    # Service constants for configuration
    SERVICE = "GoogleAI"
    SERVICE_ACCOUNT_FILE_PATH = "GOOGLE_SERVICE_ACCOUNT_FILE_PATH"
    PROJECT_ID = "GOOGLE_CLOUD_PROJECT_ID"
    CREDENTIALS_JSON = "GOOGLE_APPLICATION_CREDENTIALS_JSON"
    WORKLOAD_IDENTITY_CONFIG_PATH = "GOOGLE_WORKLOAD_IDENTITY_CONFIG_PATH"

    @staticmethod
    def get_credentials_and_project(
        secrets_manager: Any, log_func: Callable[[str], None] | None = None
    ) -> tuple[Any, str]:
        """Get Google Cloud credentials and project ID.

        Args:
            secrets_manager: GriptapeNodes.SecretsManager() instance
            log_func: Optional logging function

        Returns:
            Tuple of (credentials, project_id)

        Raises:
            ValueError: If no valid credentials are found or project ID cannot be determined
        """
        if not GOOGLE_AUTH_INSTALLED:
            raise ImportError("Google auth libraries not installed. Install 'google-auth'.")

        def _log(msg: str):
            if log_func:
                log_func(msg)

        # Get all auth-related secrets
        workload_identity_config = secrets_manager.get_secret(GoogleAuthHelper.WORKLOAD_IDENTITY_CONFIG_PATH, should_error_on_not_found=False)
        service_account_file = secrets_manager.get_secret(GoogleAuthHelper.SERVICE_ACCOUNT_FILE_PATH, should_error_on_not_found=False)
        project_id = secrets_manager.get_secret(GoogleAuthHelper.PROJECT_ID, should_error_on_not_found=False)
        credentials_json = secrets_manager.get_secret(GoogleAuthHelper.CREDENTIALS_JSON, should_error_on_not_found=False)

        credentials = None
        final_project_id = None

        # Option 1: Workload Identity Federation
        if workload_identity_config and os.path.exists(workload_identity_config):
            _log("🔑 Using workload identity federation for authentication.")
            try:
                # Use google.auth.load_credentials_from_file which auto-detects the
                # credential type (identity_pool, aws, pluggable) based on the config file
                credentials, _ = google.auth.load_credentials_from_file(
                    workload_identity_config, scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )

                # Try to extract project_id from config
                try:
                    with open(workload_identity_config) as f:
                        config = json.load(f)
                        # Try to extract from service account impersonation email
                        if "service_account_impersonation" in config:
                            sa_email = config["service_account_impersonation"].get("service_account_email", "")
                            if "@" in sa_email:
                                parts = sa_email.split("@")[1].split(".")
                                if parts:
                                    final_project_id = parts[0]
                except Exception:
                    pass

                # Fall back to environment/secret for project_id
                if not final_project_id:
                    final_project_id = project_id

                if not final_project_id:
                    raise ValueError(
                        "Could not determine project ID from workload identity config. Set GOOGLE_CLOUD_PROJECT_ID."
                    )

                _log(f"✅ Workload identity federation authentication successful for project: {final_project_id}")
                return credentials, final_project_id

            except Exception as e:
                _log(f"❌ Workload identity federation authentication failed: {e}")
                raise

        # Option 2: Service Account File
        if service_account_file and os.path.exists(service_account_file):
            _log("🔑 Using service account file for authentication.")
            try:
                with open(service_account_file) as f:
                    sa_data = json.load(f)
                    final_project_id = sa_data.get("project_id")

                credentials = service_account.Credentials.from_service_account_file(
                    service_account_file, scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )

                if not final_project_id:
                    raise ValueError("Service account file does not contain 'project_id'.")

                _log(f"✅ Service account file authentication successful for project: {final_project_id}")
                return credentials, final_project_id

            except Exception as e:
                _log(f"❌ Service account file authentication failed: {e}")
                raise

        # Option 3: Service Account JSON string
        if credentials_json:
            _log("🔑 Using JSON credentials for authentication.")
            try:
                cred_dict = json.loads(credentials_json)
                credentials = service_account.Credentials.from_service_account_info(
                    cred_dict, scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                final_project_id = cred_dict.get("project_id") or project_id

                if not final_project_id:
                    raise ValueError(
                        "Could not determine project ID. Provide GOOGLE_CLOUD_PROJECT_ID or "
                        "include 'project_id' in GOOGLE_APPLICATION_CREDENTIALS_JSON."
                    )

                _log(f"✅ JSON credentials authentication successful for project: {final_project_id}")
                return credentials, final_project_id

            except Exception as e:
                _log(f"❌ JSON credentials authentication failed: {e}")
                raise

        # Option 4: Application Default Credentials (ADC)
        # Works in production (Cloud Run, GKE, GCE) and local dev (gcloud auth)
        if project_id:
            _log("🔑 Using Application Default Credentials (Cloud Run/GKE/GCE metadata server or gcloud auth).")
            final_project_id = project_id
            # Return None for credentials to let the client libraries use ADC
            return None, final_project_id

        # No valid auth method found
        raise ValueError(
            "No credentials provided. Configure one of: "
            "GOOGLE_WORKLOAD_IDENTITY_CONFIG_PATH, "
            "GOOGLE_SERVICE_ACCOUNT_FILE_PATH, "
            "GOOGLE_APPLICATION_CREDENTIALS_JSON, or "
            "GOOGLE_CLOUD_PROJECT_ID (for ADC)."
        )

    @staticmethod
    def get_access_token(credentials: Any) -> str:
        """Get access token from credentials.

        Args:
            credentials: Google credentials object

        Returns:
            Access token string
        """
        if not credentials:
            raise ValueError("No credentials provided")

        if not credentials.valid:
            credentials.refresh(Request())

        return credentials.token


def validate_and_maybe_shrink_image(
    image_bytes: bytes,
    mime_type: str,
    image_name: str,
    allowed_mimes: set[str],
    byte_limit: int,
    auto_image_resize: bool = True,
    log_func: Callable[[str], None] | None = None,
) -> tuple[bytes, str]:
    """Validate image MIME type and size, optionally shrinking if too large.

    Args:
        image_bytes: Raw image bytes
        mime_type: MIME type of the image
        image_name: Name/identifier for error messages
        allowed_mimes: Set of allowed MIME types
        byte_limit: Maximum allowed size in bytes
        auto_image_resize: If False, fail when image exceeds limit instead of shrinking
        log_func: Optional logging function

    Returns:
        Tuple of (bytes, mime_type) - possibly converted/compressed

    Raises:
        ValueError: If MIME type is not allowed, or image is too large (strict mode or shrink failed)
    """

    def _log(msg: str):
        if log_func:
            log_func(msg)

    # Validate MIME type
    if mime_type not in allowed_mimes:
        error_msg = (
            f"❌ Image '{image_name}' has unsupported MIME type: {mime_type}. Supported: {', '.join(allowed_mimes)}"
        )
        _log(error_msg)
        raise ValueError(error_msg)

    # Check size
    if len(image_bytes) > byte_limit:
        size_mb = len(image_bytes) / (1024 * 1024)
        limit_mb = byte_limit / (1024 * 1024)

        if not auto_image_resize:
            error_msg = f"❌ Image '{image_name}' is {size_mb:.1f} MB, which exceeds the {limit_mb:.0f} MB limit. Resize the image or enable 'auto_image_resize' to allow auto-resizing."
            _log(error_msg)
            raise ValueError(error_msg)

        _log(f"ℹ️ Image '{image_name}' is {size_mb:.1f} MB; attempting to downscale to ≤ {limit_mb:.0f} MB...")
        image_bytes, mime_type = shrink_image_to_limit(image_bytes, mime_type, byte_limit, log_func=log_func)

        if len(image_bytes) <= byte_limit:
            new_mb = len(image_bytes) / (1024 * 1024)
            _log(f"✅ Downscaled '{image_name}' to {new_mb:.2f} MB ({mime_type}).")
        else:
            error_msg = f"❌ Image '{image_name}' remains too large after downscaling."
            _log(error_msg)
            raise ValueError(error_msg)

    return image_bytes, mime_type


def shrink_image_to_limit(
    image_bytes: bytes,
    mime_type: str,
    byte_limit: int,
    log_func: callable = None,
) -> tuple[bytes, str]:
    """Best-effort shrink using Pillow to ensure <= byte_limit.

    Args:
        image_bytes: Raw image bytes
        mime_type: Original MIME type of the image
        byte_limit: Maximum allowed size in bytes
        log_func: Optional logging function (e.g., self._log)

    Returns:
        Tuple of (bytes, mime_type) - possibly converted/compressed
    """

    def _log(msg: str):
        if log_func:
            log_func(msg)

    if not PIL_INSTALLED:
        _log("ℹ️ Pillow not installed; cannot downscale large images. Install 'Pillow' to enable.")
        return image_bytes, mime_type
    try:
        img = PILImage.open(_io.BytesIO(image_bytes))
        img = img.convert("RGBA") if img.mode in ("P", "LA") else img
        # Prefer WEBP for better compression and alpha support
        target_format = "WEBP"

        orig_w, orig_h = img.size

        # Try lossless first (best quality)
        buf = _io.BytesIO()
        img.save(buf, format=target_format, lossless=True, method=6)
        data = buf.getvalue()
        image_size_bytes = len(data)
        _log(f"Downscale attempt: lossless size={image_size_bytes / (1024 * 1024):.2f}MB")
        if image_size_bytes <= byte_limit:
            _log(f"Shrunk image to {image_size_bytes / (1024 * 1024):.2f}MB (lossless)")
            return data, "image/webp"

        # Finer-grained scales for better quality preservation
        scales = [1.0, 0.75, 0.5]
        qualities = [100, 95, 85]

        for scale in scales:
            w = max(1, int(orig_w * scale))
            h = max(1, int(orig_h * scale))
            resized = img.resize((w, h)) if (w, h) != (orig_w, orig_h) else img

            for q in qualities:
                buf = _io.BytesIO()
                resized.save(buf, format=target_format, quality=q, method=6)
                data = buf.getvalue()
                image_size_bytes = len(data)
                _log(f"Downscale attempt: scale={scale:.2f} quality={q} size={image_size_bytes / (1024 * 1024):.2f}MB")
                if image_size_bytes <= byte_limit:
                    _log(f"Shrunk image to {image_size_bytes / (1024 * 1024):.2f}MB (q={q})")
                    return data, "image/webp"
    except Exception as e:
        _log(f"Downscale failed: {e}")
    _log("Returning original image bytes after downscale attempts")
    return image_bytes, mime_type
