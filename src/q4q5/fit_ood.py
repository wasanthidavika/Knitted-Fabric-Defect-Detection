#!/usr/bin/env python3
"""
Fit the out-of-distribution check for the web app on TRAIN images (resized copies), report how many VALIDATION images it
would flag (should be about 1%), write outputs/q4/ood_stats.json. Test images are never used.
Usage: python src/fit_ood.py --config configs/config.yaml
"""
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ood
from common import load_config, paths


def load_rgb(p):
    im = cv2.imread(str(p))
    return im[..., ::-1] if im is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--percentile", type=float, default=99.5)
    a = ap.parse_args()
    cfg = load_config(a.config)
    P = paths(cfg)
    man = pd.read_csv(P["manifest"])
    feats = {}
    for sp in ("train", "val"):
        F = [ood.features(im) for im in (load_rgb(p) for p in man.loc[man.split == sp, "dst_path"]) if im is not None]
        feats[sp] = np.stack(F)
    st = ood.fit(feats["train"], a.percentile)
    dv = ood.distances(feats["val"], st)
    st["val_flag_rate"] = float((dv > st["limit"]).mean())
    st["n_val"] = int(len(dv))
    out = Path(cfg["outputs_dir"]) / "q4"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(st, open(out / "ood_stats.json", "w"), indent=2)
    print(f"fit on {st['n_fit']} train images; limit = {st['limit']:.2f} (train median {st['train_distance_median']:.2f}); "
          f"validation images flagged: {st['val_flag_rate']:.1%} of {st['n_val']}\n-> {out / 'ood_stats.json'}")
    print("A val flag rate far above ~1% means train and val differ in global statistics: investigate before trusting the check.")


if __name__ == "__main__":
    main()
