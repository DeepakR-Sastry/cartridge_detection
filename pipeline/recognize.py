"""
The swappable boundary.

Image in, list of (text, rectangle, confidence) out. Nothing in this file knows
about HP, cartridges, vocabularies, or XL. Swapping PaddleOCR for EasyOCR or
docTR means writing a new class with the same read() signature and changing one
line in the app.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    text: str
    box: tuple[int, int, int, int]  # left, top, right, bottom, source pixels
    confidence: float

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]


def _rect(poly) -> tuple[int, int, int, int]:
    """4-point polygon -> axis-aligned rectangle."""
    xs = [int(p[0]) for p in poly]
    ys = [int(p[1]) for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


class PaddleRecognizer:
    """
    PaddleOCR PP-OCRv6. Loads the model once, in the constructor.

    use_doc_unwarping and use_doc_orientation_classify default to True in
    PaddleOCR 3.x. Both silently transform the image and return coordinates in
    the transformed frame, so every box lands in the wrong place while the text
    reads correctly. That bug cost a day. Do not remove these flags.
    """

    def __init__(self, lang: str = "en") -> None:
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            lang=lang,
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )

    def read(self, image_path: str) -> list[Detection]:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)
        res = self._ocr.predict(image_path)[0]
        return [
            Detection(text=t, box=_rect(poly), confidence=round(float(c), 3))
            for t, c, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"])
        ]


class DumpRecognizer:
    """
    Replays a saved ocr_dump.jsonl instead of running a model. Same interface.
    Lets the eval loop and the matching tests run in a second, on CPU, with no
    GPU and no model load.
    """

    def __init__(self, dump_path: str) -> None:
        import json

        self._by_image: dict[str, list[Detection]] = {}
        with open(dump_path) as fh:
            for line in fh:
                r = json.loads(line)
                self._by_image[r["image"]] = [
                    Detection(text=d["text"], box=_rect(d["box"]), confidence=d["conf"])
                    for d in r["detections"]
                ]

    def read(self, image_path: str) -> list[Detection]:
        name = os.path.basename(image_path)
        if name not in self._by_image:
            raise KeyError(f"{name} not in dump")
        return self._by_image[name]
    