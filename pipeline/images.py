"""
Normalize an uploaded photo before anything else touches it.

Three jobs, in order:
  1. Open it, including HEIC, which PIL cannot do without pillow-heif.
  2. Apply EXIF orientation and strip the tag, so the pixels are the truth and
     no downstream code has to think about rotation again.
  3. Optionally cap the long side, since a 12MP photo over cellular is slow.

Everything downstream works in the returned pixel space. Boxes drawn on that
image and boxes returned by the recognizer are in the same coordinate frame by
construction, which is the whole point.
"""

import io
import os

from PIL import Image, ImageOps

# Registers the HEIF/HEIC opener with PIL. Import for side effect only.
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:  # pragma: no cover
    HEIC_SUPPORTED = False

# 0 means no downscale. Text is small; do not lower this without rescoring.
MAX_LONG_SIDE = int(os.environ.get("CARTRIDGE_MAX_LONG_SIDE", "0"))

JPEG_QUALITY = int(os.environ.get("CARTRIDGE_JPEG_QUALITY", "95"))


def normalize(data: bytes, max_long_side: int | None = None) -> Image.Image:
    """
    Raw upload bytes -> an RGB image in final pixel space.

    Raises ValueError if the bytes are not a readable image, which is what an
    upload endpoint wants to turn into a 400 rather than a 500.
    """
    if max_long_side is None:
        max_long_side = MAX_LONG_SIDE

    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:
        raise ValueError(f"could not read image: {e}") from e

    # exif_transpose rotates the pixels to match the orientation tag and drops
    # the tag. After this the stored pixels are the displayed pixels.
    im = ImageOps.exif_transpose(im)
    im = im.convert("RGB")

    if max_long_side and max(im.size) > max_long_side:
        scale = max_long_side / max(im.size)
        im = im.resize(
            (round(im.width * scale), round(im.height * scale)),
            Image.LANCZOS,
        )

    return im


def to_jpeg_bytes(im: Image.Image, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def save_normalized(data: bytes, path: str, max_long_side: int | None = None) -> tuple[int, int]:
    """
    Normalize and write to disk as JPEG. Returns (width, height).

    PaddleOCR takes a path, so uploads land on disk before inference. Callers
    are responsible for cleaning up; a tempfile directory is the usual answer.
    """
    im = normalize(data, max_long_side)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    im.save(path, format="JPEG", quality=JPEG_QUALITY)
    return im.size


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit("usage: python -m pipeline.images FILE [FILE ...]")

    print(f"pillow-heif: {'yes' if HEIC_SUPPORTED else 'NO, HEIC uploads will fail'}")
    for p in sys.argv[1:]:
        raw = open(p, "rb").read()
        before = Image.open(io.BytesIO(raw))
        tag = (before.getexif() or {}).get(274, 1)
        after = normalize(raw)
        flag = "  ROTATED" if before.size != after.size else ""
        print(f"{os.path.basename(p):24} exif={tag}  "
              f"{before.size[0]}x{before.size[1]} -> {after.size[0]}x{after.size[1]}{flag}")