#!/usr/bin/env python3
"""
Score the cartridge pipeline against the frozen eval set.

    python score.py --labels eval/labels --predictions runs/latest.json
    python score.py --labels eval/labels --oracle        # sanity check, should be 100%

PREDICTIONS FORMAT
  A JSON list. One entry per candidate box the pipeline drew, ranked within
  its (image, product) group by descending confidence.

    [
      {"image": "IMG_0123.jpg", "number": "67", "xl": true,
       "box": [1204, 890, 1502, 1310], "confidence": 0.91},
      ...
    ]

  Emit nothing for a query the pipeline believes is absent. Absence is a real
  answer here and is scored as one.

SCORING RULE
  A predicted box is correct if its center point falls inside a ground-truth
  box with the same number and the same XL state. No IoU threshold. The output
  is read by a human deciding which box to grab, so loose overlap is the point.

QUERY SET
  Every image is queried with every product appearing anywhere in the eval set.
  Deterministic, no sampling. Present queries feed find rate, absent queries
  feed absence accuracy.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

TOP_K = 3  # candidates drawn per query; a hit anywhere in the top K counts


def key(number, xl):
    return f"{number}{'XL' if xl else ''}"


def load_labels(labels_dir):
    gt = {}
    for fn in sorted(os.listdir(labels_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(labels_dir, fn)) as fh:
            d = json.load(fh)
        by_product = defaultdict(list)
        for b in d.get("boxes", []):
            by_product[key(b["number"], b["xl"])].append(b)
        gt[d["image"]] = by_product
    if not gt:
        sys.exit(f"no label files in {labels_dir}")
    return gt


def load_predictions(path):
    with open(path) as fh:
        rows = json.load(fh)
    preds = defaultdict(list)
    for r in rows:
        preds[(r["image"], key(r["number"], r["xl"]))].append(r)
    for v in preds.values():
        v.sort(key=lambda r: -r.get("confidence", 0.0))
    return preds


def oracle(gt):
    """Perfect predictions built from the labels. Scoring these must give 100%."""
    preds = defaultdict(list)
    for image, by_product in gt.items():
        for product, boxes in by_product.items():
            b = boxes[0]
            preds[(image, product)].append(
                {"image": image, "box": b["box"], "confidence": 1.0})
    return preds


def hits(pred_box, gt_boxes):
    cx = (pred_box[0] + pred_box[2]) / 2
    cy = (pred_box[1] + pred_box[3]) / 2
    for g in gt_boxes:
        l, t, r, b = g["box"]
        if l <= cx <= r and t <= cy <= b:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="eval/labels")
    ap.add_argument("--predictions")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--misses", action="store_true",
                    help="list every missed and every wrongly-drawn box")
    args = ap.parse_args()

    gt = load_labels(args.labels)
    if args.oracle:
        preds = oracle(gt)
    elif args.predictions:
        preds = load_predictions(args.predictions)
    else:
        sys.exit("pass --predictions FILE or --oracle")

    products = sorted({p for img in gt.values() for p in img})

    found = miss = 0                      # present queries, excluding hard-only
    hard_found = hard_miss = 0            # present queries whose instances are all hard
    top1 = 0
    absent_right = absent_wrong = 0
    boxes_drawn = boxes_wrong = 0
    missed_list, wrong_list = [], []
    per_product = defaultdict(lambda: [0, 0])

    for image in sorted(gt):
        for product in products:
            gt_boxes = gt[image].get(product, [])
            cands = preds.get((image, product), [])[:TOP_K]

            boxes_drawn += len(cands)
            bad = [c for c in cands if not hits(c["box"], gt_boxes)]
            boxes_wrong += len(bad)
            if args.misses:
                wrong_list += [(image, product, c["box"]) for c in bad]

            if not gt_boxes:
                if cands:
                    absent_wrong += 1
                else:
                    absent_right += 1
                continue

            hit = len(bad) < len(cands)
            hard_only = all(b["hard"] for b in gt_boxes)

            if hard_only:
                hard_found += hit
                hard_miss += not hit
            else:
                found += hit
                miss += not hit
                per_product[product][0] += hit
                per_product[product][1] += 1
                if not hit and args.misses:
                    missed_list.append((image, product))

            if cands and hits(cands[0]["box"], gt_boxes):
                top1 += 1

    present = found + miss
    pct = lambda n, d: f"{n/d:6.1%}" if d else "     -"

    print(f"eval set: {len(gt)} images, {len(products)} products, "
          f"{len(gt)*len(products)} queries")
    print()
    print(f"find rate        {pct(found, present)}   ({found}/{present} present queries)")
    print(f"  top-1 only     {pct(top1, present + hard_found + hard_miss)}")
    print(f"  hard only      {pct(hard_found, hard_found+hard_miss)}   "
          f"({hard_found}/{hard_found+hard_miss})")
    print(f"absence accuracy {pct(absent_right, absent_right+absent_wrong)}   "
          f"({absent_right}/{absent_right+absent_wrong} absent queries)")
    print(f"wrong box rate   {pct(boxes_wrong, boxes_drawn)}   "
          f"({boxes_wrong}/{boxes_drawn} boxes drawn)")

    worst = sorted((h/n, p, h, n) for p, (h, n) in per_product.items() if n >= 3)[:8]
    if worst:
        print("\nworst products (>=3 queries)")
        for rate, p, h, n in worst:
            print(f"  {p:<8} {rate:5.0%}  ({h}/{n})")

    if args.misses:
        if missed_list:
            print(f"\nmissed ({len(missed_list)})")
            for image, p in missed_list:
                print(f"  {image}  {p}")
        if wrong_list:
            print(f"\nwrong boxes ({len(wrong_list)})")
            for image, p, box in wrong_list:
                print(f"  {image}  {p}  {box}")


if __name__ == "__main__":
    main()