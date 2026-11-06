#!/usr/bin/env python3
"""
Run the trained classical detector end-to-end on a split (features are recomputed from the images, nothing cached),
write predictions in the shared format and measure latency.
Latency = everything after the image is decoded: Gabor/LBP maps, windows, stage 1, GLCM, SVM, regression, NMS
(single image at a time; report hardware and note that OpenCV may use several CPU threads).

Usage (development): python src/baseline_predict.py --split val --name SVM_BEST [--limit 50]
The test split is refused here; final_eval.py calls predict_split(..., allow_test=True) once.
"""
import argparse, json, sys, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_lib import extract_candidates, load_gray, windows_to_detections
from common import load_config
from metrics import PRED_COLS
from q3_utils import guard_test, hardware_string, image_table, q3_dirs


def predict_split(cfg, bundle_path, split, allow_test=False, limit=None, warmup=2):
    guard_test(cfg, split, allow_test)
    b = joblib.load(bundle_path)
    imgs = image_table(cfg)
    ids = sorted(imgs.loc[imgs["split"] == split, "image_id"])[: limit or None]
    dfs, times = [], []
    for k, iid in enumerate(ids):
        gray = load_gray(Path(cfg["resized_dir"]) / "images" / split / f"{iid}.jpg")      # decoding excluded from timing
        t0 = time.perf_counter()
        c = extract_candidates(gray, b["anchors"], b["stage1"], b["K"], b["nms_iou"])
        dfs.append(windows_to_detections(b, iid, c["boxes"], c["X"], c["hw"]))
        if k >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    preds = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=PRED_COLS)
    t = np.array(times) if times else np.array([np.nan])
    lat = {"images": len(ids), "median_ms": float(np.median(t)), "mean_ms": float(np.mean(t)), "p95_ms": float(np.percentile(t, 95)),
           "hardware": hardware_string(), "scope": "feature extraction + cascade + SVM + NMS, 1 image at a time, excludes image decoding"}
    return preds, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--name", default="SVM_BEST")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    D = q3_dirs(cfg)
    preds, lat = predict_split(cfg, a.bundle or D["baseline"] / "best.joblib", a.split, limit=a.limit)
    D["preds"].mkdir(parents=True, exist_ok=True)
    out = D["preds"] / f"{a.name}_{a.split}.csv"
    preds.to_csv(out, index=False)
    json.dump(lat, open(D["preds"] / f"{a.name}_{a.split}_latency.json", "w"), indent=2)
    print(f"{len(preds)} detections -> {out}\nlatency: {json.dumps(lat)}")


if __name__ == "__main__":
    main()
