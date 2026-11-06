"""
Fabric defect check: FastAPI backend + static frontend. Predictions are computed by the trained model at request time.

Run from the project root:   uvicorn app.main:app --host 127.0.0.1 --port 8000
Endpoints: GET /  (page) | GET /health | GET /api/info | POST /api/predict (file) | POST /api/explain (file, box, class)
"""
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import xai_core as X                                              # noqa: E402
from app.inference import AppState                                # noqa: E402
from app.validation import InputError, read_image                 # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
STATE = {}


@asynccontextmanager
async def lifespan(app):
    cfg_path = os.environ.get("ISL_APP_CONFIG", str(Path(__file__).resolve().parent / "app_config.yaml"))
    STATE["s"] = AppState(yaml.safe_load(open(cfg_path)))
    yield
    STATE.clear()


app = FastAPI(title="Knitted fabric defect check", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.exception_handler(InputError)
async def input_error_handler(request, exc):
    return JSONResponse(status_code=exc.status, content={"detail": exc.message})


@app.exception_handler(Exception)
async def unexpected_error_handler(request, exc):
    return JSONResponse(status_code=500, content={"detail": "The server could not process this image. Please try another image."})


def _image_from_upload(file):
    s = STATE["s"]
    return read_image(file.file.read(), s.limits, s.exif_mode, int(s.app_cfg.get("work_long_side", 640)))


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    s = STATE["s"]
    return {"status": "ok", "backend": s.detector.name, "model": s.app_cfg.get("model_version", ""), "warnings": s.warnings}


@app.get("/api/info")
def info():
    return STATE["s"].info()


@app.post("/api/predict")
def predict(file: UploadFile = File(...)):
    s = STATE["s"]
    img, meta = _image_from_upload(file)
    t0 = time.perf_counter()
    dets, verdict, ood = s.predict(img)                     # the trained model runs HERE, at request time
    ms = (time.perf_counter() - t0) * 1000
    return {"verdict": verdict, "detections": dets, "ood": ood, "image": meta, "threshold": s.thr, "threshold_source": s.thr_source,
            "inference_ms": round(ms, 1), "model_version": s.app_cfg.get("model_version", ""), "backend": s.detector.name,
            "preview_jpeg": X.jpeg_b64(img), "warnings": s.warnings}


@app.post("/api/explain")
def explain(file: UploadFile = File(...), x1: float = Form(...), y1: float = Form(...), x2: float = Form(...), y2: float = Form(...),
            cls: int = Form(...), methods: str = Form("occlusion,eigencam")):
    s = STATE["s"]
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise HTTPException(422, "Invalid box: coordinates must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1.")
    if cls not in s.names:
        raise HTTPException(422, "Unknown defect class.")
    wanted = [m for m in methods.split(",") if m in ("occlusion", "eigencam")]
    if not wanted:
        raise HTTPException(422, "methods must contain 'occlusion' and/or 'eigencam'.")
    img, _ = _image_from_upload(file)
    box = np.array([x1, y1, x2, y2], float)
    t0 = time.perf_counter()
    with s.lock:
        res = X.explain_detection(s.detector, img, box, cls, tuple(wanted), int(s.app_cfg.get("eigencam_layer", 15)),
                                  np.random.default_rng(0), int(s.app_cfg.get("explain_max_patches", 36)))
    out = {"score_before": res["s0"], "note": res.get("note"), "seconds": round(time.perf_counter() - t0, 2), "methods": {}}
    for m in wanted:
        d = res.get(m, {"available": False, "reason": "not computed"})
        if not d.get("available"):
            out["methods"][m] = {"available": False, "reason": d.get("reason", "")}
            continue
        entry = {"available": True, "overlay_png": X.png_b64(X.overlay(img, d["heat_norm"], box_norm=box)), "n_forward": d.get("n_forward")}
        if "drop_top" in d:
            entry.update({"hidden_window_px": list(d["window"]), "score_after_hiding_highlight": d["score_top_hidden"],
                          "drop_when_hiding_highlight": d["drop_top"], "drop_when_hiding_random": d["drop_random"], "n_random": d["n_random"]})
        out["methods"][m] = entry
    return out
