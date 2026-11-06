#!/usr/bin/env python3
"""
Q3.a learning curves + rule-based diagnosis hints for runs trained by train_yolo.py.

For each run: loss curves (train vs val: box / cls / dfl) and mAP per epoch, with the epoch where mosaic
augmentation switches off marked. Writes a diagnosis table. The hints are PROMPTS FOR YOUR ANALYSIS, not conclusions:
  * Ultralytics training losses are computed on AUGMENTED images (mosaic), so train loss > val loss is normal.
  * overfitting evidence = val loss rises while train loss keeps falling (and mAP stops improving or drops)
  * "still improving" = best mAP at the very end -> the epoch budget, not capacity, is the limit
  * underfitting / capacity evidence needs CROSS-RUN comparison (n vs s vs m, frozen backbone): compare in the report.

Usage:
  python src/learning_curves.py --config configs/config.yaml --runs E00 E01 E02
  python src/learning_curves.py --runs E00 E06 E07 --compare depth_width
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config
from q3_utils import q3_dirs, read_results_csv


def one_run(run_dir, name, figs):
    df = read_results_csv(run_dir)
    args = {}
    if (run_dir / "args.yaml").exists():
        args = yaml.safe_load(open(run_dir / "args.yaml")) or {}
    x = df["epoch"] if "epoch" in df.columns else np.arange(1, len(df) + 1)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for a_, k in zip(ax.ravel()[:3], ("box", "cls", "dfl")):
        for pre, st in (("train", "-"), ("val", "--")):
            c = f"{pre}/{k}_loss"
            if c in df.columns:
                a_.plot(x, df[c], st, label=pre)
        a_.set(title=f"{k} loss", xlabel="epoch")
        a_.legend()
    for c, lab in (("metrics/mAP50(B)", "mAP@0.5"), ("metrics/mAP50-95(B)", "mAP@0.5:0.95"),
                   ("metrics/precision(B)", "precision"), ("metrics/recall(B)", "recall")):
        if c in df.columns:
            ax[1, 1].plot(x, df[c], label=lab)
    ax[1, 1].set(title="validation metrics", xlabel="epoch")
    ax[1, 1].legend()
    cm, ep = args.get("close_mosaic"), args.get("epochs")
    if cm and ep and int(cm) < int(ep):
        for a_ in ax.ravel():
            a_.axvline(int(ep) - int(cm) + 1, color="grey", ls=":", lw=1)
        fig.text(0.5, 0.005, f"dotted line = mosaic augmentation switched off (epoch {int(ep) - int(cm) + 1})", ha="center", fontsize=8)
    fig.suptitle(f"Learning curves: {name}")
    fig.tight_layout()
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / f"learning_curve_{name}.png", dpi=130)
    plt.close(fig)

    n = len(df)
    tl = df[[c for c in df.columns if c.startswith("train/") and c.endswith("_loss")]].sum(axis=1)
    vl = df[[c for c in df.columns if c.startswith("val/") and c.endswith("_loss")]].sum(axis=1)
    m50 = df["metrics/mAP50(B)"]
    best = int(m50.idxmax())
    k = min(5, n)
    slope = float(np.polyfit(np.arange(k), m50.iloc[-k:].values, 1)[0]) if k >= 3 else float("nan")
    hints = []
    if n >= 5 and best >= n - 3 and slope > 0:
        hints.append("mAP still rising at the end: epoch budget is a limit (not converged)")
    if n >= 8 and vl.iloc[-1] > 1.1 * vl.min() and int(vl.idxmin()) < 0.7 * n:
        hints.append("val loss rises after its minimum while train loss falls: overfitting signs")
    if m50.max() < 0.02:
        hints.append("mAP50 ~ 0: training failed or labels misaligned: check label_check.png and the loss curves")
    if not np.isfinite(df.to_numpy(float)).all():
        hints.append("NaN/inf values in results: diverged")
    return {"run": name, "epochs_run": n, "best_epoch": int(x.iloc[best]), "best_mAP50": round(float(m50.max()), 4),
            "final_mAP50": round(float(m50.iloc[-1]), 4), "final_train_loss": round(float(tl.iloc[-1]), 4),
            "final_val_loss": round(float(vl.iloc[-1]), 4), "min_val_loss": round(float(vl.min()), 4),
            "min_val_loss_epoch": int(x.iloc[int(vl.idxmin())]), "mAP_last5_slope": round(slope, 5),
            "hints": "; ".join(hints) or "none triggered (still interpret the curves yourself)"}, df, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--compare", default=None, help="tag for a multi-run val mAP50 comparison figure")
    a = ap.parse_args()
    cfg = load_config(a.config)
    D = q3_dirs(cfg)
    rows, curves = [], {}
    for name in a.runs:
        run_dir = D["root"] / name
        if not (run_dir / "results.csv").exists():
            print(f"[skip] {name}: {run_dir / 'results.csv'} not found"); continue
        r, df, x = one_run(run_dir, name, D["figs"])
        rows.append(r)
        curves[name] = (x, df["metrics/mAP50(B)"])
    if not rows:
        raise SystemExit("no runs found")
    diag = pd.DataFrame(rows)
    D["tables"].mkdir(parents=True, exist_ok=True)
    diag.to_csv(D["tables"] / "learning_curve_diagnosis.csv", index=False)
    print(diag.to_string(index=False))
    if a.compare:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, (x, y) in curves.items():
            ax.plot(x, y, label=name)
        ax.set(title=f"Validation mAP@0.5 per epoch: {a.compare}", xlabel="epoch", ylabel="mAP@0.5")
        ax.legend()
        fig.tight_layout()
        fig.savefig(D["figs"] / f"compare_{a.compare}.png", dpi=130)
        plt.close(fig)
    print(f"\nFigures -> {D['figs']}\nTable   -> {D['tables'] / 'learning_curve_diagnosis.csv'}")


if __name__ == "__main__":
    main()
