#!/usr/bin/env python3
"""
Colour-shortcut check (Q1): can mean colour/brightness statistics ALONE predict which defects are present?
Trains on TRAIN, evaluates on VAL (test is never touched). If a class is predicted well above chance from
colour alone, colour is a confound to watch in error analysis.
Usage:  python src/colour_shortcut.py --config configs/config.yaml
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import load_config, paths, read_yolo_labels, set_seed


def feats(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((160, 160))
    a = np.asarray(im, dtype=np.float32) / 255.0
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255.0
    return np.r_[a.mean((0, 1)), a.std((0, 1)), hsv[..., 1].mean(), hsv[..., 1].std(), hsv[..., 2].mean(), hsv[..., 2].std()]



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    P = paths(cfg)
    names = cfg["class_names"]

    man = pd.read_csv(P["manifest"])
    man = man[man.split.isin(["train", "val"])].reset_index(drop=True)

    lab = read_yolo_labels(cfg["resized_dir"])

    # Create image-level class presence using unique image IDs.
    pres = (
        lab.drop_duplicates(["image_id", "cls"])
        .assign(v=1)
        .pivot_table(
            index="image_id",
            columns="cls",
            values="v",
            fill_value=0
        )
    )

    X = np.stack([feats(p) for p in man["dst_path"]])

    Y = pd.DataFrame(index=man["image_id"])

    for c in sorted(names):
        Y[names[c]] = (
            pres[c].reindex(man["image_id"]).fillna(0).values
            if c in pres.columns else 0
        )

    Y["ANY DEFECT"] = (Y.sum(axis=1) > 0).astype(int)

    tr = (man.split == "train").values
    va = (man.split == "val").values

    rows = []

    for col in Y.columns:
        y = Y[col].values

        if len(set(y[va])) < 2 or len(set(y[tr])) < 2:
            continue

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )

        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[va])[:, 1]

        prev = float(y[va].mean())
        ap_ = average_precision_score(y[va], p)

        rows.append({
            "target": col,
            "val_prevalence": round(prev, 3),
            "ROC_AUC": round(roc_auc_score(y[va], p), 3),
            "AP": round(ap_, 3),
            "AP_lift_vs_prevalence": round(ap_ / prev, 2)
        })

    out = pd.DataFrame(rows)

    P["q1"].mkdir(parents=True, exist_ok=True)
    table_dir = P["q1"] / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    out.to_csv(table_dir / "colour_shortcut.csv", index=False)

    print(out.to_string(index=False))

    flag = out[out.ROC_AUC > 0.65]

    print(
        "\nInterpretation: ROC_AUC 0.5 = colour carries no signal. "
        "Targets above 0.65 -> colour/brightness alone is informative:",
        list(flag.target) or "none",
        "\nReport this honestly and analyse errors per colour."
    )


if __name__ == "__main__":
    main()
