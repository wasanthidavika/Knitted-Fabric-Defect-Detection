#!/usr/bin/env python3
"""
Choose the confidence threshold on VALIDATION predictions (Q1/Q2 promise: false negatives cost more).
Picks the HIGHEST threshold whose image-level defect recall (any defect flagged) is >= the target; reports the
false-alarm rate on defect-free images at that threshold. The web app (Q4) uses this value for pass/refer.

Usage: python src/select_threshold.py --preds outputs/q3/preds/E00_val.csv --name E00 --target-recall 0.95
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
from metrics import image_level_scores
from q3_utils import load_gt, q3_dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--target-recall", type=float, default=0.95)
    a = ap.parse_args()
    cfg = load_config(a.config)
    D = q3_dirs(cfg)
    preds = pd.read_csv(a.preds)
    gt, imgs = load_gt(cfg, "val")
    cls_ids = sorted(cfg["class_names"])
    rows = []
    for t in np.round(np.arange(0.02, 0.99, 0.01), 2):
        _, s = image_level_scores(preds, gt, imgs["image_id"], cls_ids, t)
        rows.append({"conf_thr": t, "any_defect_recall": s["any_defect_recall"], "any_defect_fpr": s["any_defect_fpr"], "macro_f1": s["macro_f1"]})
    tab = pd.DataFrame(rows)
    ok = tab[tab.any_defect_recall >= a.target_recall]
    pick = ok.iloc[-1] if len(ok) else tab.iloc[0]
    res = {"name": a.name, "chosen_on": "val", "target_recall": a.target_recall, "target_reached": bool(len(ok)),
           "conf_thr": float(pick.conf_thr), "any_defect_recall": float(pick.any_defect_recall),
           "false_alarm_rate_on_defect_free_images": float(pick.any_defect_fpr), "macro_f1_at_thr": float(pick.macro_f1)}
    D["eval"].mkdir(parents=True, exist_ok=True)
    tab.to_csv(D["eval"] / f"threshold_curve_{a.name}.csv", index=False)
    json.dump(res, open(D["eval"] / f"threshold_{a.name}.json", "w"), indent=2)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tab.conf_thr, tab.any_defect_recall, label="defect recall (image level)")
    ax.plot(tab.conf_thr, tab.any_defect_fpr, label="false-alarm rate (defect-free images)")
    ax.axvline(res["conf_thr"], color="k", ls=":")
    ax.set(xlabel="confidence threshold", title=f"Threshold choice on validation: {a.name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(D["figs"] / f"threshold_{a.name}.png" if D["figs"].exists() else D["eval"] / f"threshold_{a.name}.png", dpi=120)
    print(json.dumps(res, indent=2))
    if not res["target_reached"]:
        print("WARNING: target recall not reachable at any threshold on validation.")


if __name__ == "__main__":
    main()
