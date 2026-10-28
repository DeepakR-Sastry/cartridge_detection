#!/usr/bin/env python3
"""
FastAPI wrapper around the offline pipeline.

Photos in, JSON plus annotated images out.

    python -m uvicorn serving.app:app --host 127.0.0.1 --port 8000

Environment:
    CF_RECOGNIZER   "paddle" (default) or "dump"
    CF_DUMP_PATH    dump file, required when CF_RECOGNIZER=dump
    CF_FOUND_CONF   confidence at or above which status is "found" (default 0.90)
    CF_JOB_ROOT     where per-request folders live (default runs/jobs)
    CF_JOB_TTL_MIN  delete job folders older than this (default 60)
    CF_DELIVER_MAX  longest edge of the annotated image sent back (default 1600)
"""
from fastapi.staticfiles import StaticFiles
import json
from serving import auth
import os
import re
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import pillow_heif
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps

from pipeline.draw import annotate
from pipeline.match import Candidate, Product, match
from pipeline.recognize import DumpRecognizer, PaddleRecognizer

pillow_heif.register_heif_opener()

TOP_K = int(os.environ.get("CF_TOP_K", "3"))
FOUND_CONF = float(os.environ.get("CF_FOUND_CONF", "0.90"))
JOB_ROOT = Path(os.environ.get("CF_JOB_ROOT", "runs/jobs"))
JOB_TTL = float(os.environ.get("CF_JOB_TTL_MIN", "60")) * 60
DELIVER_MAX = int(os.environ.get("CF_DELIVER_MAX", "1600"))

JOB_ID = re.compile(r"^[0-9a-f]{32}$")

STATE = {}


@dataclass(frozen=True)
class PhotoCandidate:
    """A Candidate plus which uploaded photo it came from."""
    photo: int
    candidate: Candidate


def build_recognizer():
    kind = os.environ.get("CF_RECOGNIZER", "paddle").lower()
    if kind == "dump":
        path = os.environ.get("CF_DUMP_PATH")
        if not path:
            raise RuntimeError("CF_RECOGNIZER=dump requires CF_DUMP_PATH")
        return DumpRecognizer(path)
    if kind == "paddle":
        return PaddleRecognizer()
    raise RuntimeError(f"unknown CF_RECOGNIZER: {kind}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    STATE["recognizer"] = build_recognizer()
    STATE["kind"] = os.environ.get("CF_RECOGNIZER", "paddle").lower()
    yield
    STATE.clear()


app = FastAPI(title="cartridge-finder", lifespan=lifespan)


def sweep_jobs() -> None:
    """Delete job folders past their TTL. Cheap, runs on each request."""
    cutoff = time.time() - JOB_TTL
    for d in JOB_ROOT.glob("*"):
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": STATE.get("recognizer") is not None,
        "recognizer": STATE.get("kind"),
        "found_conf": FOUND_CONF,
        "deliver_max": DELIVER_MAX,
    }


def normalize(src: Path, dst: Path) -> tuple[int, int]:
    """
    Whatever the phone sent -> the same kind of JPG the eval set uses.

    HEIC is decoded, EXIF rotation is baked into the pixels, and the result is
    saved at quality 95. Coordinates downstream are pixels on this file, which
    is the frame the eval labels are in.
    """
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        im.save(dst, "JPEG", quality=95)
        return im.size


def scan(path: Path, photo_index: int, queries: list[Product]) -> list[PhotoCandidate]:
    dets = STATE["recognizer"].read(str(path))
    hits = match(dets, queries, top_k=TOP_K)
    out = []
    for cands in hits.values():
        for c in cands:
            out.append(PhotoCandidate(photo=photo_index, candidate=c))
    return out


def parse_queries(raw: str) -> list[Product]:
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"queries is not valid JSON: {e}")
    if not isinstance(items, list) or not items:
        raise HTTPException(400, "queries must be a non-empty list")

    seen, queries = set(), []
    for item in items:
        if not isinstance(item, dict) or "number" not in item:
            raise HTTPException(400, f"bad query entry: {item!r}")
        p = Product(str(item["number"]).strip().upper(), bool(item.get("xl", False)))
        if Product.parse(str(p)) is None:
            raise HTTPException(400, f"not a product number: {item['number']!r}")
        if p not in seen:
            seen.add(p)
            queries.append(p)
    return queries


@app.post("/find")
async def find(
    photos: list[UploadFile] = File(...),
    queries: str = Form(...),
):
    wanted = parse_queries(queries)
    print(f"/find  queries={[str(p) for p in wanted]}  photos={len(photos)}", flush=True)
    if not photos:
        raise HTTPException(400, "no photos uploaded")

    sweep_jobs()
    job = uuid.uuid4().hex
    workdir = JOB_ROOT / job
    workdir.mkdir(parents=True)

    meta, found = [], {}
    for i, upload in enumerate(photos, start=1):
        raw = workdir / f"upload_{i}"
        with raw.open("wb") as fh:
            shutil.copyfileobj(upload.file, fh)

        jpg = workdir / f"photo_{i}.jpg"
        try:
            width, height = normalize(raw, jpg)
        except Exception as e:
            shutil.rmtree(workdir, ignore_errors=True)
            raise HTTPException(400, f"cannot read {upload.filename!r}: {e}")
        raw.unlink(missing_ok=True)

        meta.append({
            "index": i,
            "filename": upload.filename,
            "width": width,
            "height": height,
            "annotated_url": f"/photo/{job}/{i}",
        })

        for pc in scan(jpg, i, wanted):
            found.setdefault(pc.candidate.product, []).append(pc)

    results = []
    marks = {m["index"]: [] for m in meta}
    for p in wanted:
        # Top-k per photo, not globally. Confidence saturates at 1.0, so a
        # global sort is broken by upload order and gives every slot to the
        # first photo. If a cartridge is in two photos, both get drawn.
        by_photo: dict[int, list[PhotoCandidate]] = {}
        for pc in found.get(p, []):
            by_photo.setdefault(pc.photo, []).append(pc)

        cands = []
        for photo_index in sorted(by_photo):
            group = sorted(by_photo[photo_index],
                           key=lambda pc: -pc.candidate.confidence)
            cands.extend(group[:TOP_K])

        # Best photo first, so candidates[0] names the photo in the summary.
        cands.sort(key=lambda pc: -pc.candidate.confidence)
        if not cands:
            status = "not_found"
        elif cands[0].candidate.confidence >= FOUND_CONF:
            status = "found"
        else:
            status = "low_confidence"
        results.append({
            "number": p.number,
            "xl": p.xl,
            "status": status,
            "candidates": [
                {
                    "photo": pc.photo,
                    "box": list(pc.candidate.box),
                    "confidence": pc.candidate.confidence,
                }
                for pc in cands
            ],
        })
        for pc in cands:
            marks[pc.photo].append({
                "label": str(p),
                "box": list(pc.candidate.box),
                "rank": len(marks[pc.photo]),  # rank within this photo
                "confidence": pc.candidate.confidence,
            })

    for m in meta:
        i = m["index"]
        annotate(
            str(workdir / f"photo_{i}.jpg"),
            marks[i],
            str(workdir / f"annotated_{i}.jpg"),
            max_side=DELIVER_MAX,
        )

    return {"job": job, "photos": meta, "results": results}


@app.get("/photo/{job}/{index}")
def photo(job: str, index: int):
    if not JOB_ID.match(job):
        raise HTTPException(404, "no such photo")
    path = JOB_ROOT / job / f"annotated_{index}.jpg"
    if not path.is_file():
        raise HTTPException(404, "no such photo")
    return FileResponse(path, media_type="image/jpeg")

auth.install(app)
app.mount("/", StaticFiles(directory="static", html=True), name="static")