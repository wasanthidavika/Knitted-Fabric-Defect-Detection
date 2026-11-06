"""Input validation for uploaded images. Every failure raises InputError with an HTTP status and a plain-language message."""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class InputError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


def read_image(data, limits, exif_mode="apply", work_long_side=640):
    """bytes -> (RGB uint8 array resized to `work_long_side` on the long side, info dict)."""
    if not data:
        raise InputError(400, "The uploaded file is empty.")
    if len(data) > limits["max_upload_mb"] * 1024 * 1024:
        raise InputError(413, f"The file is larger than {limits['max_upload_mb']} MB.")
    Image.MAX_IMAGE_PIXELS = limits["max_pixels"]
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Image.DecompressionBombError:
        raise InputError(413, "The image has too many pixels to process safely.")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise InputError(415, "This file is not a readable image (JPEG, PNG, BMP, TIFF or WebP expected).")
    W, H = im.size
    if min(W, H) < limits["min_side_px"]:
        raise InputError(422, f"The image is too small ({W}x{H}); the shorter side must be at least {limits['min_side_px']} px.")
    if max(W, H) > limits["max_side_px"]:
        raise InputError(413, f"The image is too large ({W}x{H}); the longer side may be at most {limits['max_side_px']} px.")
    if max(W, H) / min(W, H) > limits["max_aspect_ratio"]:
        raise InputError(422, "The image shape is unusual (extremely long and thin); expected a normal fabric photo.")
    exif_applied = False
    if exif_mode == "apply":
        before = im.getexif().get(274, 1)
        im = ImageOps.exif_transpose(im)
        exif_applied = before != 1
    im = im.convert("RGB")
    s = work_long_side / max(im.size)
    w2, h2 = max(1, round(im.width * s)), max(1, round(im.height * s))
    im = im.resize((w2, h2), Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8), {"original_size": [W, H], "work_size": [w2, h2], "exif_rotation_applied": exif_applied}
