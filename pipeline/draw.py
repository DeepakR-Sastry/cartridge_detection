"""
Draw match rectangles on a photo.

Coordinates are pixels on the oriented JPG, the same frame the eval labels and
the /find response use. If a box lands in the wrong place, the bug is upstream
in orientation or unwarping, not here.

Boxes are drawn at full resolution, then the image is optionally downscaled for
delivery. Never downscale first: the coordinates are in source pixels.
"""

from PIL import Image, ImageDraw, ImageFont

COLORS = ["#00FF00", "#FFAA00", "#FF00FF"]  # rank 1, 2, 3


def _font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(
    image_path: str,
    marks: list[dict],
    out_path: str,
    max_side: int | None = None,
) -> None:
    """
    marks: [{"label": "902", "box": [l, t, r, b], "rank": 0, "confidence": 0.99}]
    rank 0 is the top candidate and gets the brightest colour.
    max_side: longest edge of the saved file. None keeps full resolution.
    """
    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)

    scale = max(im.width, im.height) / 4000
    width = max(3, int(6 * scale))
    text_size = max(14, int(48 * scale))
    font = _font(text_size)

    for m in marks:
        left, top, right, bottom = m["box"]
        colour = COLORS[min(m.get("rank", 0), len(COLORS) - 1)]

        pad = width * 2
        draw.rectangle(
            [left - pad, top - pad, right + pad, bottom + pad],
            outline=colour,
            width=width,
        )

        label = m["label"]
        if "confidence" in m:
            label += f" {m['confidence']:.2f}"

        tl, tt, tr, tb = draw.textbbox((0, 0), label, font=font)
        th = tb - tt
        ty = top - pad - th - width * 2
        if ty < 0:
            ty = bottom + pad + width
        draw.rectangle(
            [left - pad, ty, left - pad + (tr - tl) + width * 2, ty + th + width * 2],
            fill=colour,
        )
        draw.text((left - pad + width, ty + width - tt), label, fill="#000000", font=font)

    if max_side and max(im.width, im.height) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)

    im.save(out_path, "JPEG", quality=90)