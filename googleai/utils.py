"""Common utilities for GoogleAI nodes."""

try:
    import io as _io

    from PIL import Image as PILImage

    PIL_INSTALLED = True
except Exception:
    PIL_INSTALLED = False


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
