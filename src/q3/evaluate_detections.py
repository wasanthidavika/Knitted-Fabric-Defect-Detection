#!/usr/bin/env python3
"""
Score a predictions CSV against ground truth with the shared metric code (src/metrics.py).
Reports mAP@0.5 (PRIMARY), mAP@0.5:0.95, per-class AP@0.5, image-level macro-F1 / defect recall at a confidence
threshold, and mAP@0.5 per fabric colour (grey vs dyed, estimated colour).

Usage (development): python src/evaluate_detections.py --preds outputs/q3/preds/E00_val.csv --split val --name E00
"""
import argparse, json, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config
from metrics import evaluate_detections, image_level_scores
from q3_utils import guard_test, load_gt, q3_dirs, resolve_conf_thr


def evaluate_split(cfg, preds, split, conf_thr=0.25, allow_test=False):
    guard_test(cfg, split, allow_test)
    gt, imgs = load_gt(cfg, split)
    names = cfg["class_names"]
    cls_ids = sorted(names)
    res = evaluate_detections(preds, gt, cls_ids)
    out = {"split": split, "n_images": int(len(imgs)), "n_gt_boxes": int(len(gt)),
           "map50": res["map50"], "map50_95": res["map50_95"],
           "ap50_per_class": {names[c]: v for c, v in res["ap50_per_class"].items()},
           "n_gt_per_class": {names[c]: v for c, v in res["n_gt_per_class"].items()}}
    _, il = image_level_scores(preds, gt, imgs["image_id"], cls_ids, conf_thr)
    out["image_level"] = il
    by_col = {}
    for col in ("grey", "dyed"):
        ids = set(imgs.loc[imgs["colour"] == col, "image_id"])
        if ids:
            r = evaluate_detections(preds[preds.image_id.isin(ids)], gt[gt.image_id.isin(ids)], cls_ids)
            by_col[col] = {"images": len(ids), "map50": r["map50"], "map50_95": r["map50_95"]}
    out["by_colour"] = by_col
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--compare-log-id", default=None, help="YOLO run id: print its Ultralytics-logged val mAP50 next to ours")
    ap.add_argument("--conf-thr", type=float, default=None, help="default 0.25 unless --threshold-json is given")
    ap.add_argument("--threshold-json", default=None, help="output of select_threshold.py (threshold chosen on VAL)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    a.conf_thr = resolve_conf_thr(a.conf_thr, a.threshold_json)
    D = q3_dirs(cfg)
    out = evaluate_split(cfg, pd.read_csv(a.preds), a.split, a.conf_thr)
    D["eval"].mkdir(parents=True, exist_ok=True)
    json.dump(out, open(D["eval"] / f"{a.name}_{a.split}.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
    if a.compare_log_id and a.split == "val":
        from q3_utils import log_path
        lp = log_path(cfg)
        if lp.exists():
            row = pd.read_csv(lp, dtype=str, keep_default_na=False).set_index("id")
            if a.compare_log_id in row.index and row.loc[a.compare_log_id, "val_mAP50"]:
                u = float(row.loc[a.compare_log_id, "val_mAP50"])
                print(f"\nmAP@0.5 on val: shared metrics.py = {out['map50']:.4f} | Ultralytics (logged) = {u:.4f} | difference = {out['map50'] - u:+.4f}"
                      "\n-> report this difference; large gaps (> ~0.02) mean something differs (image size, conf, NMS IoU).")


if __name__ == "__main__":
    main()
