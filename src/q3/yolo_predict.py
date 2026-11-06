#!/usr/bin/env python3
"""
Run a trained YOLO model over every image of a split -> predictions CSV in the shared metric format, plus latency.

Uses conf=0.001 (needed for a proper mAP) and the same NMS IoU as Ultralytics validation (0.7).
Latency = preprocess + inference + postprocess per image at batch size 1 (image decoding excluded), first images
discarded as warm-up. Report the hardware string with it.

Usage (development): python src/yolo_predict.py --weights outputs/q3/E00/weights/best.pt --split val --name E00
The test split is refused here; final_eval.py calls predict_split(..., allow_test=True) exactly once.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config
from metrics import PRED_COLS
from q3_utils import guard_test, hardware_string, image_table, q3_dirs


def predict_split(cfg, weights, split, conf=0.001, iou=0.7, imgsz=640, max_det=300, device=None, warmup=5, allow_test=False):
    guard_test(cfg, split, allow_test)
    from ultralytics import YOLO
    imgs = image_table(cfg)
    ids = imgs.loc[imgs["split"] == split, "image_id"].tolist()
    model = YOLO(str(weights))
    rows, times = [], []
    for k, iid in enumerate(ids):
        p = Path(cfg["resized_dir"]) / "images" / split / f"{iid}.jpg"
        r = model.predict(str(p), conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device=device, verbose=False)[0]
        if k >= warmup:
            times.append(r.speed["preprocess"] + r.speed["inference"] + r.speed["postprocess"])
        b = r.boxes
        if b is not None and len(b):
            xy, c, s = b.xyxyn.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()
            rows += [(iid, int(ci), float(si), *map(float, bx)) for ci, si, bx in zip(c, s, xy)]
    preds = pd.DataFrame(rows, columns=PRED_COLS)
    t = np.array(times) if times else np.array([np.nan])
    lat = {"images": len(ids), "median_ms": float(np.median(t)), "mean_ms": float(np.mean(t)),
           "p95_ms": float(np.percentile(t, 95)), "hardware": hardware_string(), "imgsz": imgsz,
           "scope": "preprocess+inference+postprocess, batch 1, excludes image decoding"}
    return preds, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    D = q3_dirs(cfg)
    preds, lat = predict_split(cfg, a.weights, a.split, imgsz=a.imgsz, device=a.device)
    D["preds"].mkdir(parents=True, exist_ok=True)
    out = D["preds"] / f"{a.name}_{a.split}.csv"
    preds.to_csv(out, index=False)
    json.dump(lat, open(D["preds"] / f"{a.name}_{a.split}_latency.json", "w"), indent=2)
    print(f"{len(preds)} detections -> {out}\nlatency: {json.dumps(lat)}")


if __name__ == "__main__":
    main()
