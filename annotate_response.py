#!/usr/bin/env python3
"""
Draw a /find response onto the photos it came from.

    curl ... > runs/resp.json
    python annotate_response.py runs/resp.json eval/images runs/annotated
"""

import json
import os
import sys

from pipeline.draw import annotate


def main():
    resp_path, images_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(resp_path) as fh:
        resp = json.load(fh)

    by_photo = {p["index"]: p["filename"] for p in resp["photos"]}
    marks = {i: [] for i in by_photo}

    for r in resp["results"]:
        label = r["number"] + ("XL" if r["xl"] else "")
        for rank, c in enumerate(r["candidates"]):
            marks[c["photo"]].append({
                "label": label,
                "box": c["box"],
                "rank": rank,
                "confidence": c["confidence"],
            })

    os.makedirs(out_dir, exist_ok=True)
    for index, filename in by_photo.items():
        out = os.path.join(out_dir, f"photo{index}_{filename}")
        annotate(os.path.join(images_dir, filename), marks[index], out)
        print(f"{len(marks[index])} boxes -> {out}")


if __name__ == "__main__":
    main()