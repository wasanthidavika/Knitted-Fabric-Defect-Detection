#!/usr/bin/env python3
"""Fake ISL-Knit-like dataset WITH visible class-specific defect patterns, so the classical baseline can learn something.
(Pipeline check only: these numbers mean nothing about the real data.)"""
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

rng = np.random.default_rng(1)
root = Path("smoke_data/Full")
shutil.rmtree("smoke_data", ignore_errors=True)
P = np.array([.08, .06, .08, .40, .14, .14, .10])


def draw(arr, c, x1, y1, x2, y2):
    reg = arr[y1:y2, x1:x2].astype(np.int16)
    h, w = reg.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    if c == 0: reg -= 75
    elif c == 1: reg += 75
    elif c == 2: reg += (60 * (((yy // 3) % 2) * 2 - 1))[..., None]
    elif c == 3: reg += (60 * (((xx // 3) % 2) * 2 - 1))[..., None]
    elif c == 4: reg += (60 * ((((xx // 4) + (yy // 4)) % 2) * 2 - 1))[..., None]
    elif c == 5:
        m = (yy < 3) | (yy >= h - 3) | (xx < 3) | (xx >= w - 3); reg[m] -= 90
    else: reg += (np.linspace(-80, 80, w)[None, :, None] * np.ones((h, 1, 1))).astype(np.int16)
    arr[y1:y2, x1:x2] = reg.clip(0, 255)


for sp, n in [("train", 150), ("valid", 40), ("test", 40)]:
    (root / sp / "images").mkdir(parents=True)
    (root / sp / "labels").mkdir(parents=True)
    for i in range(n):
        dyed = rng.random() < 0.15
        W, H = [(1280, 960), (960, 1280), (1200, 900)][rng.integers(0, 3)]
        base = rng.integers(60, 230, 3) if dyed else np.full(3, rng.integers(100, 160))
        arr = (base + rng.normal(0, 14, (H, W, 3))).clip(0, 255)
        low = Image.fromarray(rng.normal(128, 25, (6, 8, 3)).clip(0, 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
        arr = (arr + np.asarray(low, float) - 128).clip(0, 255).astype(np.int16)
        if not dyed:
            arr[:] = arr.mean(2, keepdims=True).astype(np.int16)
        lines = []
        for _ in range(rng.integers(0, 7)):
            c = int(rng.choice(7, p=P))
            bw, bh = int(rng.integers(30, 120)), int(rng.integers(30, 120))
            x1, y1 = int(rng.integers(0, W - bw)), int(rng.integers(0, H - bh))
            draw(arr, c, x1, y1, x1 + bw, y1 + bh)
            lines.append(f"{c} {(x1 + bw / 2) / W:.5f} {(y1 + bh / 2) / H:.5f} {bw / W:.5f} {bh / H:.5f}")
        stem = f"{'dyed' if dyed else 'grey'}_{sp}_{i}"
        Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(root / sp / "images" / f"{stem}.jpg", quality=92)
        (root / sp / "labels" / f"{stem}.txt").write_text("\n".join(lines))
print("synthetic Q3 dataset written to", root)
