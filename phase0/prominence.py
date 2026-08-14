import json, os, re, collections, statistics

DUMP = os.path.expanduser("~/cartridge_detection/phase0/ocr_dump.jsonl")
NUMLIKE = re.compile(r"^\s*(\d{2,4})\s*(XL|XI|X1|xl)?\s*$", re.I)

def height(box):
    ys = [p[1] for p in box]
    return max(ys) - min(ys)

rel = collections.defaultdict(list)

for line in open(DUMP):
    r = json.loads(line)
    hits = [(d["text"].strip().upper(), height(d["box"]))
            for d in r["detections"] if NUMLIKE.match(d["text"])]
    if not hits:
        continue
    ref = max(h for _, h in hits)          # tallest number-like text in this image
    for t, h in hits:
        rel[t].append(round(h / ref, 2))

rows = [(t, len(v), round(statistics.median(v), 2)) for t, v in rel.items() if len(v) >= 2]
rows.sort(key=lambda x: x[2])

print(f"{'token':>8} {'n':>4} {'median relative height':>8}")
for t, n, m in rows:
    print(f"{t:>8} {n:>4} {m:>8}")