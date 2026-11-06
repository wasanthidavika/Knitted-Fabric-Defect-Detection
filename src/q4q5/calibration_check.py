#!/usr/bin/env python3
"""
Empirical reliability of detection scores on VALIDATION data (Q4: "do not describe an uncalibrated score as a guaranteed
confidence"). For each score bin: how often were detections of that score correct (IoU >= 0.5, right class)?
Writes outputs/q3/eval/calibration_<name>.json (the web app shows it next to each score) and a reliability plot.

Usage: python src/calibration_check.py --preds outputs/q3/preds/F42_val.csv --name F42 --floor 0.05
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config
from error_analysis import match
from q3_utils import load_gt, q3_dirs


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0, c - h)), float(min(1, c + h)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--floor", type=float, default=0.05)
    ap.add_argument("--bins", type=int, default=10)
    a = ap.parse_args()
    cfg = load_config(a.config)
    D = q3_dirs(cfg)
    gt, imgs = load_gt(cfg, "val")
    pr, _ = match(pd.read_csv(a.preds), gt, a.floor)
    edges = np.linspace(0, 1, a.bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = pr[(pr.conf >= lo) & ((pr.conf < hi) if hi < 1 else (pr.conf <= hi))]
        k, n = int((s.status == "TP").sum()), len(s)
        ci = wilson(k, n)
        rows.append({"lo": round(float(lo), 3), "hi": round(float(hi), 3), "n": n, "n_correct": k,
                     "precision": (k / n) if n else None, "ci_lo": ci[0], "ci_hi": ci[1]})
    res = {"name": a.name, "chosen_on": "val", "n_val_images": int(len(imgs)), "conf_floor": a.floor, "iou": 0.5,
           "definition": "share of detections in the score bin that matched a true defect of the same class (IoU >= 0.5)",
           "bins": rows}
    D["eval"].mkdir(parents=True, exist_ok=True)
    json.dump(res, open(D["eval"] / f"calibration_{a.name}.json", "w"), indent=2)
    t = pd.DataFrame(rows)
    print(t.round(3).to_string(index=False))
    ok = t.dropna(subset=["precision"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], "k:", label="perfect calibration")
    ax.errorbar((ok.lo + ok.hi) / 2, ok.precision, yerr=[ok.precision - ok.ci_lo, ok.ci_hi - ok.precision], fmt="o-", label="validation precision (95% CI)")
    ax.set(xlabel="detector score", ylabel="share of detections that were correct", title=f"Reliability on validation: {a.name}")
    ax.legend()
    fig.tight_layout()
    out = Path(cfg["outputs_dir"]) / "q4" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"reliability_{a.name}.png", dpi=130)
    print(f"-> {D['eval'] / ('calibration_' + a.name + '.json')}  and  {out / ('reliability_' + a.name + '.png')}")
    print("Say in the report whether the curve is above/below the diagonal: the score is a ranking, not a probability.")


if __name__ == "__main__":
    main()
