import json, os, re, collections

DUMP = os.path.expanduser("~/cartridge_detection/phase0/ocr_dump.jsonl")
NUMLIKE = re.compile(r"^\s*(\d{2,4})\s*(XL|XI|X1|xl)?\s*$", re.I)

per_image = {}
tokens = collections.Counter()
heights = []

for line in open(DUMP):
    r = json.loads(line)
    hits = []
    for d in r["detections"]:
        m = NUMLIKE.match(d["text"])
        if not m:
            continue
        ys = [p[1] for p in d["box"]]
        h = max(ys) - min(ys)
        hits.append((d["text"].strip(), d["conf"], h))
        tokens[d["text"].strip().upper()] += 1
        heights.append(h)
    per_image[r["image"]] = hits

print("=== number-like tokens, most common ===")
for t, n in tokens.most_common(60):
    print(f"{n:4d}  {t}")

print("\n=== per image: count of number-like detections ===")
for img, hits in per_image.items():
    print(f"{img}: {len(hits)}")

print("\n=== text heights (px) ===")
heights.sort()
if heights:
    for q in (0, 25, 50, 75, 100):
        print(f"p{q}: {heights[min(len(heights)-1, q*len(heights)//100)]}")