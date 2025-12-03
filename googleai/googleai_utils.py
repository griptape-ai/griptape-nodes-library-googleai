"""Common utilities for GoogleAI nodes."""

from typing import Callable

try:
    import io as _io

    from PIL import Image as PILImage

    PIL_INSTALLED = True
except Exception:
    PIL_INSTALLED = False


def validate_and_maybe_shrink_image(
    image_bytes: bytes,
    mime_type: str,
    image_name: str,
    allowed_mimes: set[str],
    byte_limit: int,
    strict_size: bool = False,
    log_func: Callable[[str], None] | None = None,
) -> tuple[bytes, str]:
    """Validate image MIME type and size, optionally shrinking if too large.

    Args:
        image_bytes: Raw image bytes
        mime_type: MIME type of the image
        image_name: Name/identifier for error messages
        allowed_mimes: Set of allowed MIME types
        byte_limit: Maximum allowed size in bytes
        strict_size: If True, fail when image exceeds limit instead of shrinking
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
        error_msg = f"❌ Image '{image_name}' has unsupported MIME type: {mime_type}. Supported: {', '.join(allowed_mimes)}"
        _log(error_msg)
        raise ValueError(error_msg)

    # Check size
    if len(image_bytes) > byte_limit:
        size_mb = len(image_bytes) / (1024 * 1024)
        limit_mb = byte_limit / (1024 * 1024)

        if strict_size:
            error_msg = f"❌ Image '{image_name}' is {size_mb:.1f} MB, which exceeds the {limit_mb:.0f} MB limit. Resize the image or disable 'strict_image_size' to allow auto-shrinking."
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
        target_mime = "image/webp"

        orig_w, orig_h = img.size
        scales = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        qualities = [90, 85, 80, 75, 70, 60, 50]

        for scale in scales:
            w = max(1, int(orig_w * scale))
            h = max(1, int(orig_h * scale))
            resized = img.resize((w, h)) if (w, h) != (orig_w, orig_h) else img
            for q in qualities:
                buf = _io.BytesIO()
                save_params = {"format": target_format, "quality": q}
                # lossless false by default; ensure efficient encoding
                if target_format == "WEBP":
                    save_params.update({"method": 6})
                resized.save(buf, **save_params)
                data = buf.getvalue()
                if len(data) <= byte_limit:
                    return data, target_mime
        # As a last resort, try JPEG without alpha
        rgb = img.convert("RGB")
        for scale in scales:
            w = max(1, int(orig_w * scale))
            h = max(1, int(orig_h * scale))
            resized = rgb.resize((w, h)) if (w, h) != (orig_w, orig_h) else rgb
            for q in qualities:
                buf = _io.BytesIO()
                resized.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
                data = buf.getvalue()
                if len(data) <= byte_limit:
                    return data, "image/jpeg"
    except Exception as e:
        _log(f"⚠️ Downscale failed: {e}")
    return image_bytes, mime_type
