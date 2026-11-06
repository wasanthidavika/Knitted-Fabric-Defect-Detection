#!/usr/bin/env python3
"""
CLIP label audit (Q6, adjacent task "labelling / label auditing"): find annotations whose appearance disagrees with their class.

For every ground-truth box of TRAIN and VAL (never test): crop with context -> CLIP image embedding -> nearest neighbours among
the other crops (excluding crops of the SAME image) -> class-prior-balanced neighbour vote. A box is flagged when the neighbours
mostly belong to another class and rarely to its own. Flags are CANDIDATES for human review, not verdicts; labels are never changed
(the split and all experiments stay valid). To report the audit honestly, review about 50 flagged and 50 random crops yourself and
state how many flags were real label problems (precision) and what that means for the error analysis.

Usage: python src/clip_label_audit.py --config configs/config.yaml [--k 15]    (Q6_FAKE=1 for a pipeline test)
"""
import argparse, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_zeroshot import get_scorer, load_img
from common import load_config
from q3_utils import load_gt


def crop_with_context(img, box_norm, context=3.0, min_side=64, out=224):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = np.asarray(box_norm) * [W, H, W, H]
    side = int(max(context * max(x2 - x1, y2 - y1), min_side))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = int(np.clip(cx - side / 2, 0, max(0, W - side))), int(np.clip(cy - side / 2, 0, max(0, H - side)))
    c = img[b:b + side, a:a + side]
    return cv2.resize(c, (out, out), interpolation=cv2.INTER_AREA if c.shape[0] > out else cv2.INTER_LINEAR)


def knn_vote(E, labels, image_ids, class_ids, k=15, chunk=2048):
    """Class-prior-balanced neighbour vote, excluding neighbours from the same image. Returns (n, C) distribution."""
    n, C = len(E), len(class_ids)
    pos = {c: i for i, c in enumerate(class_ids)}
    lab = np.array([pos[c] for c in labels])
    prior = np.maximum(np.bincount(lab, minlength=C) / n, 1e-9)
    img = pd.factorize(pd.Series(image_ids))[0]
    out = np.zeros((n, C))
    k = min(k, max(1, n - 1))
    for s in range(0, n, chunk):
        sim = E[s:s + chunk] @ E.T
        same = img[s:s + chunk, None] == img[None, :]
        sim[same] = -np.inf
        idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        for r in range(len(idx)):
            ok = np.isfinite(sim[r, idx[r]])
            votes = np.bincount(lab[idx[r][ok]], minlength=C) / prior
            out[s + r] = votes / votes.sum() if votes.sum() > 0 else 1.0 / C
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--context", type=float, default=3.0)
    ap.add_argument("--flag-p-label", type=float, default=0.2, help="flag if neighbours give the labelled class less than this share")
    ap.add_argument("--flag-p-pred", type=float, default=0.5, help="...and another class at least this share")
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--max-crops", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    names = cfg["class_names"]
    class_ids = sorted(names)
    rows, crops, cache = [], [], {}
    for split in ("train", "val"):                                   # test boxes are never touched
        gt, _ = load_gt(cfg, split)
        for r in gt.itertuples():
            rows.append({"split": split, "image_id": r.image_id, "label": int(r.cls), "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2})
    df = pd.DataFrame(rows)
    if a.max_crops:
        df = df.sample(min(a.max_crops, len(df)), random_state=cfg["seed"]).reset_index(drop=True)
    for (split, iid), g in df.groupby(["split", "image_id"]):
        img = load_img(cfg, split, iid)
        for i in g.index:
            crops.append((i, crop_with_context(img, df.loc[i, ["x1", "y1", "x2", "y2"]].values.astype(float), a.context)))
    order = [i for i, _ in crops]
    scorer = get_scorer(a.model)
    E = np.concatenate([scorer.encode_images([c for _, c in crops[s:s + 256]]) for s in range(0, len(crops), 256)])
    df = df.loc[order].reset_index(drop=True)
    P = knn_vote(E, df["label"].values, df["image_id"].values, class_ids, a.k)
    df["knn_pred"] = [class_ids[j] for j in P.argmax(1)]
    df["p_label"] = [P[i, class_ids.index(l)] for i, l in enumerate(df["label"])]
    df["p_pred"] = P.max(1)
    df["flag"] = (df["knn_pred"] != df["label"]) & (df["p_label"] < a.flag_p_label) & (df["p_pred"] >= a.flag_p_pred)
    df["label_name"], df["knn_name"] = df["label"].map(names), df["knn_pred"].map(names)
    out = Path(cfg["outputs_dir"]) / "q6"
    out.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["x1", "y1", "x2", "y2"]).to_csv(out / "label_audit.csv", index=False)
    summ = df.groupby("label_name").agg(boxes=("flag", "size"), flagged=("flag", "sum")).assign(flag_rate=lambda t: (t.flagged / t.boxes).round(3))
    summ.to_csv(out / "label_audit_summary.csv")
    top = df[df.flag].assign(margin=lambda t: t.p_pred - t.p_label).sort_values("margin", ascending=False).head(a.top)
    if len(top):
        fig, ax = plt.subplots((len(top) + 5) // 6, 6, figsize=(12, 2.2 * ((len(top) + 5) // 6)), squeeze=False)
        for a_ in ax.ravel():
            a_.axis("off")
        for a_, (k, r) in zip(ax.ravel(), top.iterrows()):
            a_.imshow(crops[k][1])
            a_.set_title(f"label: {r.label_name}\nkNN: {r.knn_name}", fontsize=7)
        fig.suptitle("Most suspicious labels (candidates for human review)")
        fig.tight_layout()
        fig.savefig(out / "label_audit_top.png", dpi=120)
    print(summ.to_string())
    print(f"\n{int(df.flag.sum())} of {len(df)} boxes flagged ({df.flag.mean():.1%}). -> {out}")
    print("Review ~50 flagged + ~50 random crops by eye and report the precision of the flags. Labels were NOT changed.")


if __name__ == "__main__":
    main()
