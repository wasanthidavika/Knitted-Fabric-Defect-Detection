#!/usr/bin/env python3
"""
Q1 exploratory data analysis. Produces tables, figures and an auto-written findings.md.

Integrity rule: analyses that drive modelling decisions use TRAIN+VAL only. Test images appear only in
descriptive counts (class_distribution.csv, dataset_description.csv).

Needs: prepare_data.py (manifest + resized dataset) and audit_isl_knit.py (proposed_splits.csv, report.json).
Usage:  python src/eda.py --config configs/config.yaml [--examples]
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common import load_config, paths, read_yolo_labels, SPLITS, DEV_SPLITS

RES_LIST = [224, 384, 512, 640, 960, 1280]
TINY = 8          # px: below this a defect is assumed unresolvable by a CNN


def short_side(df, R):
    """Short side (px) of each box if its image is resized so the LONG side equals R (aspect kept)."""
    scale = R / np.maximum(df["W"], df["H"])
    return np.minimum(df["w"] * df["W"], df["h"] * df["H"]) * scale



def load_all(cfg):
    P = paths(cfg)

    man = pd.read_csv(P["manifest"])
    sp = pd.read_csv(P["split_csv"])

    # Create unique image IDs using the same method as prepare_data.py.
    import hashlib

    def make_image_id(src):
        relative_path = Path(src).resolve().relative_to(
            Path(cfg["raw_root"]).resolve()
        ).as_posix()
        return hashlib.sha256(relative_path.encode()).hexdigest()[:16]

    sp["image_id"] = sp["path"].map(make_image_id)

    # Match manifest records with the frozen split information.
    imgs = man.merge(
        sp[["image_id", "colour", "group", "has_defect"]],
        on="image_id",
        how="left",
        validate="one_to_one"
    )

    boxes = read_yolo_labels(cfg["resized_dir"])

    # Check that every label belongs to a known image.
    orphan = set(boxes["image_id"]) - set(imgs["image_id"])

    if orphan:
        raise SystemExit(
            f"{len(orphan)} label files have no entry in manifest.csv "
            f"(e.g. {sorted(orphan)[:3]}). "
            "Re-run prepare_data.py without --limit."
        )

    # Attach image dimensions to each bounding box.
    boxes = boxes.merge(
        imgs[["image_id", "W", "H"]],
        on="image_id",
        how="left",
        validate="many_to_one"
    )

    names = cfg["class_names"]

    boxes["cls_name"] = boxes["cls"].map(names)
    boxes["short_px_640"] = short_side(boxes, 640)
    boxes["area"] = boxes["w"] * boxes["h"]
    boxes["aspect"] = (
        (boxes["w"] * boxes["W"]) /
        (boxes["h"] * boxes["H"])
    )

    # Count bounding boxes per unique image.
    box_counts = boxes.groupby("image_id").size()
    imgs["n_boxes"] = (
        imgs["image_id"].map(box_counts).fillna(0).astype(int)
    )

    # Image-level class presence matrix.
    pres = (
        boxes.groupby(["image_id", "cls"])
        .size()
        .unstack(fill_value=0)
        .gt(0)
    )

    pres = pres.reindex(
        index=imgs["image_id"],
        columns=sorted(names),
        fill_value=False
    )

    pres.index = imgs.index

    return P, imgs, boxes, pres, names


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--examples", action="store_true", help="also build per-class example crops")
    a = ap.parse_args()
    cfg = load_config(a.config)
    P, imgs, boxes, pres, names = load_all(cfg)
    T, F = P["q1"] / "tables", P["q1"] / "figures"
    T.mkdir(parents=True, exist_ok=True)
    F.mkdir(parents=True, exist_ok=True)
    cls_ids = sorted(names)
    cname = [names[i] for i in cls_ids]

    dev_i = imgs[imgs.split.isin(DEV_SPLITS)]
    dev_b = boxes[boxes.split.isin(DEV_SPLITS)]
    dev_p = pres.loc[dev_i.index]

    # ---- 1. dataset description (counts only, may include test) -------------------------------
    desc = imgs.groupby("split").agg(
        images=("image_id", "size"),
        no_defect=("n_boxes", lambda s: int((s == 0).sum())),
        mean_W=("W", "mean"),
        mean_H=("H", "mean")
    ).reindex(SPLITS)
    desc["boxes"] = boxes.groupby("split").size().reindex(SPLITS)
    desc.round(1).to_csv(T / "dataset_description.csv")
    size_counts = Counter(zip(imgs.W, imgs.H)).most_common(5)
    summary = {"images": len(imgs), "boxes": len(boxes), "classes": len(cls_ids),
               "no_defect_images": int((imgs.n_boxes == 0).sum()), "top_image_sizes": {f"{w}x{h}": n for (w, h), n in size_counts},
               "width_range": [int(imgs.W.min()), int(imgs.W.max())]}

    # ---- 2. class distribution ------------------------------------------------------------------
    inst = boxes.pivot_table(index="cls", columns="split", values="cx", aggfunc="size", fill_value=0)
    inst = inst.reindex(index=cls_ids, columns=SPLITS, fill_value=0)
    inst["total"] = inst.sum(axis=1)
    inst["share_%"] = (100 * inst["total"] / inst["total"].sum()).round(1)
    inst.index = cname
    inst.to_csv(T / "class_instances.csv")
    imgs_pc = pd.DataFrame({sp: pres[imgs.split == sp].sum() for sp in SPLITS})
    imgs_pc["total"] = imgs_pc.sum(axis=1)
    imgs_pc.index = cname
    imgs_pc.to_csv(T / "class_images.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    inst[SPLITS].plot(kind="bar", ax=ax[0], logy=True, title="Instances per class (log scale)")
    imgs_pc[SPLITS].plot(kind="bar", ax=ax[1], title="Images containing each class")
    savefig(fig, F / "class_distribution.png")

    # ---- 3. boxes per image (dev) ---------------------------------------------------------------
    nb = dev_i[dev_i.n_boxes > 0].n_boxes
    nb.describe().to_csv(T / "boxes_per_image.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(nb.clip(upper=40), bins=40)
    ax.set(title="Boxes per defective image (train+val, clipped at 40)", xlabel="boxes", ylabel="images")
    savefig(fig, F / "boxes_per_image.png")

    # ---- 4. box size (dev) ----------------------------------------------------------------------
    q = dev_b.groupby("cls_name")["short_px_640"].describe(percentiles=[.05, .25, .5, .75, .95]).reindex(cname)
    q.round(2).to_csv(T / "box_short_side_px_at640.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot([dev_b[dev_b.cls == c]["short_px_640"] for c in cls_ids], showfliers=False)
    ax.set_xticks(range(1, len(cname) + 1), cname)
    ax.set(yscale="log", ylabel="short side (px) at long side 640", title="Defect size by class (train+val)")
    ax.axhline(TINY, color="r", ls="--", lw=1)
    plt.setp(ax.get_xticklabels(), rotation=25)
    savefig(fig, F / "box_size_by_class.png")

    # ---- 5. resolution vs defect visibility (dev) -----------------------------------------------
    rows = []
    for R in RES_LIST:
        s = short_side(dev_b, R)
        row = {"long_side_px": R, "median_short_px": round(float(s.median()), 2),
               "share_below_8px_%": round(100 * float((s < TINY).mean()), 1)}
        for c in cls_ids:
            row[f"<8px {names[c]} %"] = round(100 * float((s[dev_b.cls == c] < TINY).mean()), 1)
        rows.append(row)
    res_tab = pd.DataFrame(rows)
    res_tab.to_csv(T / "resolution_vs_visibility.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(res_tab.long_side_px, res_tab["share_below_8px_%"], "k-o", lw=2, label="all")
    for c in cls_ids:
        ax.plot(res_tab.long_side_px, res_tab[f"<8px {names[c]} %"], alpha=.6, label=names[c])
    ax.set(xlabel="input long side (px)", ylabel="% of defects with short side < 8 px", title="Resolution vs defect visibility (train+val)")
    ax.legend(fontsize=7)
    savefig(fig, F / "resolution_vs_visibility.png")

    # ---- 6. co-occurrence (dev) -----------------------------------------------------------------
    Mx = dev_p.astype(int).T.values @ dev_p.astype(int).values
    cond = Mx / np.maximum(np.diag(Mx)[:, None], 1)
    pd.DataFrame(Mx, index=cname, columns=cname).to_csv(T / "cooccurrence_counts.csv")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cond, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(cname)), cname, rotation=40, ha="right")
    ax.set_yticks(range(len(cname)), cname)
    for i in range(len(cname)):
        for j in range(len(cname)):
            ax.text(j, i, f"{cond[i, j]:.2f}", ha="center", va="center", color="w", fontsize=7)
    ax.set_title("P(column class present | row class present)")
    fig.colorbar(im)
    savefig(fig, F / "cooccurrence.png")

    # ---- 7. position heatmap (dev) --------------------------------------------------------------
    H2, _, _ = np.histogram2d(dev_b.cx, 1 - dev_b.cy, bins=20, range=[[0, 1], [0, 1]])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(H2.T[::-1], extent=[0, 1, 0, 1], aspect="auto", cmap="magma")
    ax.set(title="Defect-centre positions (train+val)", xlabel="x (normalised)", ylabel="y (normalised)")
    savefig(fig, F / "position_heatmap.png")

    # ---- 8. image sizes -------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(imgs.W, imgs.H, s=8, alpha=.3)
    ax[0].set(xlabel="width px", ylabel="height px", title="Original image sizes (after EXIF handling)")
    ax[1].hist(imgs.W / imgs.H, bins=40)
    ax[1].set(xlabel="aspect (W/H)", title="Aspect ratio")
    savefig(fig, F / "image_sizes.png")

    # ---- 9. colour (dev) ------------------------------------------------------------------------
    col = pd.DataFrame({c: dev_p[c].groupby(dev_i["colour"]).sum() for c in cls_ids}).T
    col.index = cname
    col["dyed_share_%"] = (100 * col.get("dyed", 0) / col.sum(axis=1)).round(1)
    col.to_csv(T / "class_by_colour.csv")
    imgs.groupby(["split", "colour"]).size().unstack(fill_value=0).to_csv(T / "colour_by_split.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    col.drop(columns="dyed_share_%").plot(kind="bar", stacked=True, ax=ax, title="Images per class: grey vs dyed (estimated, train+val)")
    savefig(fig, F / "class_by_colour.png")

    # ---- 10. optional example crops -------------------------------------------------------------
    if a.examples:
        n = 6
        fig, axs = plt.subplots(len(cls_ids), n, figsize=(n * 1.6, len(cls_ids) * 1.6))
        for r, c in enumerate(cls_ids):
            sub = dev_b[dev_b.cls == c].sample(min(n, (dev_b.cls == c).sum()), random_state=cfg["seed"])
            for k in range(n):
                axs[r, k].axis("off")
                if k >= len(sub):
                    continue
                b = sub.iloc[k]
                spl = imgs.loc[imgs.image_id == b.image_id, "split"].iloc[0]
                im = Image.open(
                    Path(cfg["resized_dir"]) / "images" / spl / f"{b.image_id}.jpg"
                ).convert("RGB")
                bw, bh = b.w * im.width, b.h * im.height
                half = max(3 * max(bw, bh), 48) / 2
                cx, cy = b.cx * im.width, b.cy * im.height
                axs[r, k].imshow(im.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half))).resize((128, 128)))
                if k == 0:
                    axs[r, k].set_title(names[c], fontsize=7, loc="left")
        savefig(fig, F / "examples_per_class.png")

    # ---- 11. findings.md (auto) -----------------------------------------------------------------
    s640, s224 = res_tab.set_index("long_side_px").loc[640], res_tab.set_index("long_side_px").loc[224]
    s384, s512 = res_tab.set_index("long_side_px").loc[384], res_tab.set_index("long_side_px").loc[512]
    top, bot = inst["total"].idxmax(), inst["total"].idxmin()
    dyed = col["dyed_share_%"]
    audit = json.load(open(P["audit_report"])) if P["audit_report"].exists() else {}
    md = f"""# Q1 findings that constrain modelling (auto-generated from train+val statistics)

## Finding 1 - Defects are small
Median defect short side is {s224.median_short_px} px at long side 224 and {s640.median_short_px} px at 640.
Share of defects below {TINY} px: {s224['share_below_8px_%']}% at 224, {s384['share_below_8px_%']}% at 384, {s512['share_below_8px_%']}% at 512, {s640['share_below_8px_%']}% at 640.
**Consequence:** a 224 px classifier would erase most defects; the detector uses 640 px (test 960 as an experiment); metrics use IoU 0.5 because strict IoU on 10-20 px boxes mostly measures annotation noise.

## Finding 2 - Imbalance and many instances per image
{top} holds {inst.loc[top, 'share_%']}% of instances, {bot} only {inst.loc[bot, 'share_%']}%. Defective images carry a median of {nb.median():.0f} boxes (mean {nb.mean():.1f}).
**Consequence:** accuracy is meaningless; report per-class AP and macro metrics; consider class-loss gain/oversampling as experiments; per-class results for rare classes will be noisy.

## Finding 3 - Leakage and colour confounding
Cross-split near-duplicate pairs found by the audit: {audit.get('near_dup_pairs_cross_split', 'n/a')}; exact-duplicate groups crossing splits: {audit.get('exact_dup_groups_cross_split', 'n/a')}.
Dyed share (estimated) ranges from {dyed.min()}% to {dyed.max()}% across classes.
**Consequence:** group-aware frozen split (see split_verification.json); colour-only shortcut check (colour_shortcut.py); results reported per colour.

## Dataset summary
{json.dumps(summary, indent=2)}
"""
    (P["q1"] / "findings.md").write_text(md)
    print(md)
    print(f"\nTables -> {T}\nFigures -> {F}")


if __name__ == "__main__":
    main()
