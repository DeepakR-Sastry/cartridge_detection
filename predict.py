#!/usr/bin/env python3
"""
Run the pipeline over the eval set and write predictions for score.py.

Replay a saved dump (fast, CPU, no model load):
    python predict.py --dump phase0/ocr_dump.jsonl --out runs/baseline.json

Run the real model over the eval images:
    python predict.py --images eval/images --out runs/live.json

Both paths go through the same match() call, so the replay result is a valid
check on the live one.
"""

import argparse
import json
import os

from pipeline.match import Product, match
from pipeline.recognize import DumpRecognizer, PaddleRecognizer


def load_eval(labels_dir):
    """Eval image names, and the vocabulary standing in for the query dropdown."""
    images, vocab = [], set()
    for fn in sorted(os.listdir(labels_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(labels_dir, fn)) as fh:
            d = json.load(fh)
        images.append(d["image"])
        for b in d["boxes"]:
            vocab.add(Product(b["number"], b["xl"]))
    return images, sorted(vocab, key=lambda p: (p.number, p.xl))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="eval/labels")
    ap.add_argument("--images", default="eval/images")
    ap.add_argument("--dump", help="replay this dump instead of running the model")
    ap.add_argument("--out", default="runs/latest.json")
    args = ap.parse_args()

    images, vocab = load_eval(args.labels)
    print(f"{len(images)} images, {len(vocab)} products in vocabulary")

    rec = DumpRecognizer(args.dump) if args.dump else PaddleRecognizer()

    rows = []
    for i, name in enumerate(images, 1):
        dets = rec.read(os.path.join(args.images, name))
        hits = match(dets, vocab)
        for p, cands in hits.items():
            for c in cands:
                rows.append({
                    "image": name,
                    "number": p.number,
                    "xl": p.xl,
                    "box": list(c.box),
                    "confidence": c.confidence,
                })
        print(f"{i}/{len(images)} {name}  {len(dets)} dets  "
              f"{len(hits)} products  {sum(len(v) for v in hits.values())} boxes",
              flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\n{len(rows)} predictions -> {args.out}")


if __name__ == "__main__":
    main()