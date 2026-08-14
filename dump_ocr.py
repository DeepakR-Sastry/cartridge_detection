import glob, json, os, time
from paddleocr import PaddleOCR

SRC = os.path.expanduser("~/cartridge_detection/jpg")
OUT = os.path.expanduser("~/cartridge_detection/phase0/ocr_dump.jsonl")

ocr = PaddleOCR(use_textline_orientation=True, lang="en")
paths = sorted(glob.glob(SRC + "/*.jpg"))

with open(OUT, "w") as f:
    for i, p in enumerate(paths, 1):
        t0 = time.time()
        res = ocr.predict(p)[0]
        rec = {
            "image": os.path.basename(p),
            "seconds": round(time.time() - t0, 2),
            "detections": [
                {"text": t, "conf": round(float(c), 3), "box": [[int(x), int(y)] for x, y in poly]}
                for t, c, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"])
            ],
        }
        f.write(json.dumps(rec) + "\n")
        print(f"{i}/{len(paths)} {rec['image']} {len(rec['detections'])} dets {rec['seconds']}s", flush=True)