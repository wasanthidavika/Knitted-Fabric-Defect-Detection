#!/usr/bin/env python3
"""
Setup step: materialise the frozen split as a resized YOLO dataset.

 * reads  <audit_dir>/proposed_splits.csv   (from src/audit_isl_knit.py)
 * resizes every image to `img_long_side` px on the long side (aspect ratio kept, so the
   normalised YOLO labels stay valid) and writes  images/<split>/ + labels/<split>/
 * writes manifest.csv (original size AFTER EXIF handling, EXIF orientation, resized size)
 * writes data.yaml (train+val only) and data_test.yaml (adds test; use ONLY in final_eval)
 * --check draws boxes on sample images -> outputs/q1/label_check.png (verify EXIF handling!)

Usage:  python src/prepare_data.py --config configs/config.yaml --check
"""
import argparse, shutil, sys, hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from common import load_config, paths, set_seed

PALETTE = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0"]


def label_path_for(src):
    src = Path(src)

    parts = list(src.parts)

    try:
        i = parts.index("images")
    except ValueError:
        return None

    parts[i] = "labels"

    return Path(*parts).with_suffix(".txt")

def make_image_id(src, raw_root):
    src = Path(src).resolve()
    raw_root = Path(raw_root).resolve()

    relative_path = src.relative_to(raw_root).as_posix()

    return hashlib.sha256(relative_path.encode()).hexdigest()[:16]

def _job(a):
    src, dst, long_side, exif_mode = a
    try:
        with Image.open(src) as im:
            orient = int(im.getexif().get(274, 1))
            if exif_mode == "apply":
                im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            W, H = im.size
            s = long_side / max(W, H)
            rim = im.resize((max(1, round(W * s)), max(1, round(H * s))), Image.LANCZOS)
        rim.save(dst, quality=95)
        return {"ok": True, "W": W, "H": H, "exif_orientation": orient, "rW": rim.width, "rH": rim.height}
    except Exception as e:
        return {"ok": False, "err": str(e)[:120]}


def draw_label_check(cfg, manifest, n=24):
    P = paths(cfg)
    prefer = manifest[manifest.exif_orientation.isin([6, 8])]
    pick = pd.concat([prefer.sample(len(prefer), random_state=cfg["seed"]),
                      manifest.sample(min(n, len(manifest)), random_state=cfg["seed"])]).drop_duplicates("image_id").head(n)
    tiles = []
    for r in pick.itertuples():
        img = Image.open(Path(cfg["resized_dir"]) / "images" / r.split / f"{r.image_id}.jpg").convert("RGB")
        d = ImageDraw.Draw(img)
        lf = Path(cfg["resized_dir"]) / "labels" / r.split / f"{r.image_id}.txt"
        for ln in (lf.read_text().splitlines() if lf.exists() else []):
            t = ln.split()
            if len(t) != 5:
                continue
            c, cx, cy, w, h = int(float(t[0])), *map(float, t[1:])
            x0, y0, x1, y1 = (cx - w / 2) * img.width, (cy - h / 2) * img.height, (cx + w / 2) * img.width, (cy + h / 2) * img.height
            d.rectangle([x0, y0, x1, y1], outline=PALETTE[c % len(PALETTE)], width=3)
        d.text((6, 6), f"{r.stem[:28]} exif={r.exif_orientation}", fill="white")
        img.thumbnail((300, 300))
        tiles.append(img)
    cols = 6
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 300, rows * 300), "black")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 300, (i // cols) * 300))
    P["q1"].mkdir(parents=True, exist_ok=True)
    sheet.save(P["q1"] / "label_check.png")
    print(f"Wrote {P['q1'] / 'label_check.png'}  -> open it: do the boxes sit on the defects (esp. images with exif != 1)?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="TESTING ONLY: process N random images (overwrites manifest.csv!)")
    ap.add_argument("--check", action="store_true", help="draw boxes on sample images")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    P = paths(cfg)
    sp = pd.read_csv(P["split_csv"])
    if a.limit:
        sp = sp.sample(a.limit, random_state=cfg["seed"])
    res = Path(cfg["resized_dir"])
    for sub in ("images", "labels"):            # start clean: stale files from an older split would silently corrupt results
        shutil.rmtree(res / sub, ignore_errors=True)
    jobs, meta = [], []
    for r in sp.itertuples():
        src = Path(r.path)
        (res / "images" / r.proposed_split).mkdir(parents=True, exist_ok=True)
        (res / "labels" / r.proposed_split).mkdir(parents=True, exist_ok=True)
        image_id = make_image_id(src, cfg["raw_root"])

        dst = res / "images" / r.proposed_split / f"{image_id}.jpg"
        lbl = label_path_for(src)
        out_l = res / "labels" / r.proposed_split / f"{image_id}.txt"
        if lbl is not None and lbl.exists():
            out_l.write_text(lbl.read_text())
        else:
            print("WARNING: no label file for", src.name, "-> writing empty label")
            out_l.write_text("")
        jobs.append((str(src), str(dst), cfg["img_long_side"], cfg["exif_mode"]))
        meta.append({
            "image_id": image_id,
            "stem": src.stem,
            "split": r.proposed_split,
            "src_path": str(src),
            "dst_path": str(dst)
        })

        
    print(f"Resizing {len(jobs)} images to long side {cfg['img_long_side']} (exif_mode={cfg['exif_mode']}) ...")
    with ProcessPoolExecutor(a.workers) as ex:
        out = list(ex.map(_job, jobs, chunksize=8))
    man = pd.concat([pd.DataFrame(meta), pd.DataFrame(out)], axis=1)
    bad = man[~man["ok"].astype(bool)]
    if len(bad):
        print(f"WARNING: {len(bad)} images failed:\n", bad[["stem", "err"]].head())
    man = man[man["ok"].astype(bool)].drop(columns=["ok", "err"], errors="ignore")
    man.to_csv(P["manifest"], index=False)

    names = cfg["class_names"]
    base = {"path": str(res.resolve()), "train": "images/train", "val": "images/val", "names": names}
    yaml.safe_dump(base, open(res / "data.yaml", "w"), sort_keys=False)
    yaml.safe_dump({**base, "test": "images/test"}, open(res / "data_test.yaml", "w"), sort_keys=False)

    print("\nImages per split:", man.split.value_counts().to_dict())
    print("EXIF orientation values:", man.exif_orientation.value_counts().to_dict())
    if (man.exif_orientation != 1).any():
        print("NOTE: some images carry EXIF rotation. Use --check and confirm boxes line up; "
              "if they do NOT, set exif_mode: ignore in the config and re-run.")
    print("Wrote", P["manifest"], ",", res / "data.yaml", ",", res / "data_test.yaml")
    if a.check:
        draw_label_check(cfg, man)


if __name__ == "__main__":
    main()
