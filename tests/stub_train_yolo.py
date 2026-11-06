#!/usr/bin/env python3
"""Stand-in for train_yolo.py used ONLY to test run_experiments.py / learning_curves.py without Ultralytics.
Writes an Ultralytics-style results.csv (with padded column names) and prints 'Run directory: ...'."""
import argparse, sys
from pathlib import Path
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config"); ap.add_argument("--name"); ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--seed", type=int, default=42)
a, _ = ap.parse_known_args()
cfg = load_config(a.config)
run = Path(cfg["outputs_dir"]) / "q3" / a.name
run.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(a.seed + len(a.name))
n = a.epochs
ep = np.arange(1, n + 1)
t = np.exp(-ep / 10)
cols = ["epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss", "metrics/precision(B)", "metrics/recall(B)",
        "metrics/mAP50(B)", "metrics/mAP50-95(B)", "val/box_loss", "val/cls_loss", "val/dfl_loss"]
m50 = 0.55 * (1 - t) + rng.normal(0, 0.005, n)
vl = 1.0 + t + (0.03 * np.maximum(ep - 12, 0) if a.name == "E10" else 0)       # E10: val loss rises -> overfitting hint
tl = 1.2 * t + 0.3
if a.name == "E03":                                                               # E03: diverged
    tl = tl.copy(); tl[8:] = np.nan; m50 = np.full(n, 0.001)
rows = np.stack([ep, tl, tl * .7, tl * .9, m50 + .1, m50 + .05, m50, m50 * .5, vl, vl * .7, vl * .9], 1)
with open(run / "results.csv", "w") as f:
    f.write(",".join(f"{c:>20}" for c in cols) + "\n")
    for r in rows:
        f.write(",".join(f"{v:>20.6f}" for v in r) + "\n")
yaml.safe_dump({"epochs": n, "close_mosaic": 5}, open(run / "args.yaml", "w"))
print(f"Run directory: {run}")
