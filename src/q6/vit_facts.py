#!/usr/bin/env python3
"""
Facts for the transformer explanation (the lecturer's 5-mark point: anchor tokens/sequence length to YOUR data and resolution).
Pure arithmetic + your own box statistics: no torch needed. Writes outputs/q6/vit_facts.{json,md}.

Usage: python src/vit_facts.py --config configs/config.yaml
"""
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def n_tokens(img, patch):
    return (img // patch) ** 2


def attention_entries(tokens_with_cls):
    return tokens_with_cls ** 2


def vit_params(img, patch=16, dim=768, depth=12, mlp=3072, classes=1000):
    """Analytic parameter count of a standard ViT (pre-norm, learned positional embeddings, CLS token, linear head)."""
    n = n_tokens(img, patch) + 1
    patch_embed = 3 * patch * patch * dim + dim
    block = 2 * 2 * dim + (dim * 3 * dim + 3 * dim) + (dim * dim + dim) + (dim * mlp + mlp) + (mlp * dim + dim)
    return patch_embed + dim + n * dim + depth * block + 2 * dim + (dim * classes + classes)


def letterbox_square(img, size, pad_value=114):
    """Resize the long side to `size`, keep aspect ratio, pad to size x size (defect shapes are not distorted)."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    out = np.full((size, size) + img.shape[2:], pad_value, img.dtype)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    from common import load_config
    from q3_utils import load_gt
    cfg = load_config(a.config)
    gt, imgs = load_gt(cfg, "train")                         # TRAIN boxes only
    long_side = int(cfg.get("img_long_side", 640))
    rows = []
    for name, img, patch in (("ViT-B/16 @384", 384, 16), ("ViT-B/16 @224", 224, 16), ("CLIP ViT-B/32 @224", 224, 32)):
        short = gt["short_px"] * img / long_side              # defect short side (px) at that input size, letterboxed
        tok = n_tokens(img, patch)
        rows.append({"model": name, "input": f"{img}x{img}", "patch": patch, "patch_tokens": tok, "sequence_length_with_CLS": tok + 1,
                     "attention_entries_per_head_per_layer": attention_entries(tok + 1),
                     "median_defect_short_side_px": round(float(short.median()), 1),
                     "share_defects_smaller_than_one_patch_%": round(100 * float((short < patch).mean()), 1),
                     "params_with_7_class_head": vit_params(img, patch, classes=7) if "ViT-B/16" in name else None})
    out = Path(cfg["outputs_dir"]) / "q6"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out / "vit_facts.json", "w"), indent=2)
    hdr = list(rows[0].keys())
    md = "| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" + "\n".join("| " + " | ".join(str(r[h]) for h in hdr) + " |" for r in rows)
    (out / "vit_facts.md").write_text(md)
    print(md)
    print("\nUse these in the report: tokens/sequence length at your input size, the cost of attention (n^2), and how many defects are smaller than one patch.")


if __name__ == "__main__":
    main()
