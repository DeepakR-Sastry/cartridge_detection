#!/usr/bin/env python3
"""
Every detection on one image, with box and confidence. No matching, no
vocabulary. Answers "did the recognizer read this at all".

    python peek.py runs/jobs/<job>/photo_1.jpg
    python peek.py runs/jobs/<job>/photo_1.jpg 910
"""

import sys

from pipeline.recognize import PaddleRecognizer


def main():
    path = sys.argv[1]
    needle = sys.argv[2].upper() if len(sys.argv) > 2 else None

    dets = PaddleRecognizer().read(path)
    print(f"{len(dets)} detections\n")

    for d in sorted(dets, key=lambda d: (d.box[1], d.box[0])):
        if needle and needle not in d.text.upper():
            continue
        l, t, r, b = d.box
        print(f"{d.text!r:>14}  conf {d.confidence:.3f}  "
              f"h={b - t:>4}  at ({l},{t})")


if __name__ == "__main__":
    main()