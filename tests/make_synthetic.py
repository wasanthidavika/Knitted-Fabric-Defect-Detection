#!/usr/bin/env python3
"""Creates a tiny fake ISL-Knit-like dataset (YOLO layout) to smoke-test the pipeline without real data.
Includes: grey/dyed images, EXIF-rotated images, a planted cross-split duplicate, imbalanced classes."""
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

rng = np.random.default_rng(0)
root = Path("smoke_data/Full")
shutil.rmtree("smoke_data", ignore_errors=True)
class_p = np.array([.05, .03, .05, .5, .15, .15, .07])
for sp, n in [("train", 120), ("valid", 30), ("test", 30)]:
    (root / sp / "images").mkdir(parents=True)
    (root / sp / "labels").mkdir(parents=True)
    for i in range(n):
        dyed = rng.random() < 0.15
        W, H = [(1200, 800), (800, 1200), (1000, 750)][rng.integers(0, 3)]
        base = rng.integers(40, 255, 3) if dyed else np.full(3, rng.integers(90, 170))
        arr = (base + rng.normal(0, 18, (H, W, 3))).clip(0, 255).astype(np.uint8)
        low = Image.fromarray(rng.normal(128, 40, (6, 8, 3)).clip(0, 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
        arr = (arr.astype(np.int16) + np.asarray(low, dtype=np.int16) - 128).clip(0, 255).astype(np.uint8)   # per-image structure
        if not dyed:
            arr[:] = arr.mean(2, keepdims=True).astype(np.uint8)
        stem = f"{'dyed' if dyed else 'grey'}_{sp}_{i}"
        im = Image.fromarray(arr)
        exif = Image.Exif()
        if rng.random() < 0.15:
            exif[274] = 6                       # EXIF rotation tag
        im.save(root / sp / "images" / f"{stem}.jpg", exif=exif)
        lines = []
        for _ in range(rng.integers(0, 8)):
            c = rng.choice(7, p=class_p)
            lines.append(f"{c} {rng.uniform(.1, .9):.4f} {rng.uniform(.1, .9):.4f} {rng.uniform(.01, .08):.4f} {rng.uniform(.01, .08):.4f}")
        (root / sp / "labels" / f"{stem}.txt").write_text("\n".join(lines))
for i in range(4):                                # planted leak: train images copied into test
    stem = [p for p in (root / "train" / "images").glob("*.jpg")][i].stem
    shutil.copy(root / "train/images" / f"{stem}.jpg", root / "test/images" / f"{stem}_copy.jpg")
    shutil.copy(root / "train/labels" / f"{stem}.txt", root / "test/labels" / f"{stem}_copy.txt")
(root / "data.yaml").write_text("names: [Loop Mark, Needle Mark, Set Up, Oil Spot, Hole, Contamination, Yarn Missing]\n")
print("synthetic dataset written to", root)
