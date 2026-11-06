#!/usr/bin/env python3
"""
Q3.a error analysis of a model from its saved predictions (no new inference).

Matching at a confidence threshold, IoU >= 0.5, per image: same-class matches first (greedy by confidence), then
remaining predictions that overlap a remaining ground-truth box are CLASS CONFUSIONS, leftovers are FALSE POSITIVES
(on background) and unmatched ground truth are MISSED (false negatives).

Outputs in outputs/q3/error_analysis/<name>_<split>/ :
  confusion_matrix.csv/.png, per_class.csv, recall_by_size.csv (defect size in px), recall_by_colour.csv,
  recall_by_crowding.csv (defects per image), image_level.json, worst_cases.csv, worst_fn.png, worst_fp.png, summary.md
Run it on VAL while developing. For the TEST split it only runs after final_eval.py has written the lock file
(it reads saved predictions, so the test set is not evaluated a second time).

Usage: python src/error_analysis.py --preds outputs/q3/preds/E00_val.csv --split val --name E00 --conf-thr 0.25
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config
from metrics import PRED_COLS, image_level_scores, iou_matrix
from q3_utils import load_gt, q3_dirs, resolve_conf_thr

SIZE_BINS = [(0, 8, "<8 px"), (8, 16, "8-16 px"), (16, 32, "16-32 px"), (32, 1e9, ">=32 px")]


def match(preds, gt, conf_thr, iou_thr=0.5):
    pr = preds[preds["conf"] >= conf_thr].sort_values("conf", ascending=False).reset_index(drop=True)
    gt = gt.reset_index(drop=True)
    pstat, pgt = np.array(["FP"] * len(pr), dtype=object), np.full(len(pr), -1)
    gstat, gpred = np.array(["MISSED"] * len(gt), dtype=object), np.full(len(gt), -1)
    gidx = {k: v.values for k, v in gt.groupby("image_id").groups.items()} if len(gt) else {}
    pidx = {k: v.values for k, v in pr.groupby("image_id").groups.items()} if len(pr) else {}
    pb, pc = pr[["x1", "y1", "x2", "y2"]].values, pr["cls"].values
    gb, gc = gt[["x1", "y1", "x2", "y2"]].values, gt["cls"].values
    for img, pi in pidx.items():
        gi = gidx.get(img)
        if gi is None:
            continue
        iou = iou_matrix(pb[pi], gb[gi])
        taken = np.zeros(len(gi), bool)
        for passno in (1, 2):
            for a, p in enumerate(pi):
                if pstat[p] != "FP":
                    continue
                ok = ~taken & (iou[a] >= iou_thr)
                if passno == 1:
                    ok &= gc[gi] == pc[p]
                if ok.any():
                    j = int(np.argmax(np.where(ok, iou[a], -1)))
                    taken[j] = True
                    same = gc[gi[j]] == pc[p]
                    pstat[p] = "TP" if same else "CONFUSION"
                    pgt[p] = gc[gi[j]]
                    gstat[gi[j]] = "TP" if same else "CONFUSION"
                    gpred[gi[j]] = pc[p]
    pr["status"], pr["gt_cls"] = pstat, pgt
    gt["status"], gt["pred_cls"] = gstat, gpred
    return pr, gt


def contact_sheet(rows, preds_thr, gt, cfg, split, path, title, n=12):
    names = cfg["class_names"]
    tiles = []
    for iid in rows:
        img = Image.open(Path(cfg["resized_dir"]) / "images" / split / f"{iid}.jpg").convert("RGB")
        d = ImageDraw.Draw(img)
        for r in gt[gt.image_id == iid].itertuples():
            d.rectangle([r.x1 * img.width, r.y1 * img.height, r.x2 * img.width, r.y2 * img.height], outline="#00ff00", width=2)
        for r in preds_thr[preds_thr.image_id == iid].sort_values("conf", ascending=False).head(8).itertuples():
            d.rectangle([r.x1 * img.width, r.y1 * img.height, r.x2 * img.width, r.y2 * img.height], outline="#ff0000", width=2)
            d.text((r.x1 * img.width, max(0, r.y1 * img.height - 10)), f"{names.get(int(r.cls), r.cls)[:6]} {r.conf:.2f}", fill="#ff5555")
        img.thumbnail((360, 360))
        tiles.append(img)
    if not tiles:
        return
    cols = 4
    fig, ax = plt.subplots((len(tiles) + cols - 1) // cols, cols, figsize=(cols * 3.4, ((len(tiles) + cols - 1) // cols) * 3.1), squeeze=False)
    for a_ in ax.ravel():
        a_.axis("off")
    for a_, t, iid in zip(ax.ravel(), tiles, rows):
        a_.imshow(t)
        a_.set_title(iid[:10], fontsize=7)
    fig.suptitle(f"{title}  (green = ground truth, red = 8 most confident predictions)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--conf-thr", type=float, default=None, help="default 0.25 unless --threshold-json is given")
    ap.add_argument("--threshold-json", default=None, help="output of select_threshold.py (threshold chosen on VAL)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    a.conf_thr = resolve_conf_thr(a.conf_thr, a.threshold_json)
    D = q3_dirs(cfg)
    if a.split == "test" and not D["lock"].exists():
        raise SystemExit("Test error analysis is only allowed AFTER final_eval.py has run (lock file missing).")
    names = cfg["class_names"]
    cls_ids = sorted(names)
    preds = pd.read_csv(a.preds)
    gt, imgs = load_gt(cfg, a.split)
    pr, g = match(preds, gt, a.conf_thr)
    out = D["err"] / f"{a.name}_{a.split}"
    out.mkdir(parents=True, exist_ok=True)

    # confusion matrix
    labels_r = [names[c] for c in cls_ids] + ["background"]
    labels_c = [names[c] for c in cls_ids] + ["missed"]
    pos = {c: i for i, c in enumerate(cls_ids)}
    M = np.zeros((len(labels_r), len(labels_c)), int)
    for r in g.itertuples():
        M[pos[r.cls], pos[r.pred_cls] if r.pred_cls >= 0 else len(cls_ids)] += 1
    for r in pr[pr.status == "FP"].itertuples():
        M[len(cls_ids), pos[r.cls]] += 1
    cm = pd.DataFrame(M, index=labels_r, columns=labels_c)
    cm.to_csv(out / "confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(np.log1p(M), cmap="Blues")
    ax.set_xticks(range(len(labels_c)), labels_c, rotation=40, ha="right")
    ax.set_yticks(range(len(labels_r)), labels_r)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j]:
                ax.text(j, i, M[i, j], ha="center", va="center", fontsize=7)
    ax.set(xlabel="predicted", ylabel="true", title=f"Detection confusion at conf>={a.conf_thr}, IoU>=0.5")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    # per class
    rows = []
    for c in cls_ids:
        gi, pi = g[g.cls == c], pr[pr.cls == c]
        tp = int((gi.status == "TP").sum())
        rows.append({"class": names[c], "gt_boxes": len(gi), "recall": tp / max(len(gi), 1),
                     "precision": tp / max(len(pi), 1), "missed": int((gi.status == "MISSED").sum()),
                     "confused_as_other_class": int((gi.status == "CONFUSION").sum()),
                     "false_positives_on_background": int(((pi.status == "FP")).sum())})
    pd.DataFrame(rows).round(3).to_csv(out / "per_class.csv", index=False)

    # subgroups
    def recall_table(col, bins=None, labels=None):
        res = []
        for lab, mask in (labels if labels is not None else []):
            s = g[mask]
            res.append({"group": lab, "gt_boxes": len(s), "recall_class_correct": round(float((s.status == "TP").mean()), 3) if len(s) else None,
                        "detected_any_class": round(float(s.status.isin(["TP", "CONFUSION"]).mean()), 3) if len(s) else None})
        return pd.DataFrame(res)
    recall_table("size", labels=[(lab, (g.short_px >= lo) & (g.short_px < hi)) for lo, hi, lab in SIZE_BINS]).to_csv(out / "recall_by_size.csv", index=False)
    colour_rows = recall_table("colour", labels=[(c_, g.colour == c_) for c_ in ("grey", "dyed")])
    img_col = imgs.set_index("image_id")["colour"]
    fp_img = pr[pr.status == "FP"].groupby("image_id").size()
    colour_rows["images"] = [int((imgs.colour == c_).sum()) for c_ in ("grey", "dyed")]
    colour_rows["false_positives_per_image"] = [round(float(fp_img.reindex(imgs.loc[imgs.colour == c_, "image_id"]).fillna(0).mean()), 3) for c_ in ("grey", "dyed")]
    colour_rows.to_csv(out / "recall_by_colour.csv", index=False)
    n_in = g.groupby("image_id")["cls"].transform("size")
    recall_table("crowding", labels=[(lab, (n_in >= lo) & (n_in <= hi)) for lo, hi, lab in [(1, 1, "1 defect"), (2, 5, "2-5"), (6, 15, "6-15"), (16, 10**6, ">=16")]]).to_csv(out / "recall_by_crowding.csv", index=False)

    # image level
    _, il = image_level_scores(preds, gt, imgs["image_id"], cls_ids, a.conf_thr)
    json.dump(il, open(out / "image_level.json", "w"), indent=2)

    # worst cases
    fn = g[g.status == "MISSED"].groupby("image_id").size().sort_values(ascending=False)
    fpn = pr[pr.status == "FP"].groupby("image_id").size().sort_values(ascending=False)
    worst = pd.concat([fn.rename("missed_defects"), fpn.rename("false_positives")], axis=1).fillna(0).astype(int)
    worst = worst.join(imgs.set_index("image_id")[["colour"]])
    worst.sort_values(["missed_defects", "false_positives"], ascending=False).to_csv(out / "worst_cases.csv")
    contact_sheet(list(fn.index[:12]), pr, gt, cfg, a.split, out / "worst_fn.png", "Images with most MISSED defects")
    contact_sheet(list(fpn.index[:12]), pr, gt, cfg, a.split, out / "worst_fp.png", "Images with most FALSE POSITIVES")

    n_gt = len(g)
    md = (f"# Error analysis: {a.name} on {a.split} (conf >= {a.conf_thr}, IoU >= 0.5)\n\n"
          f"- ground-truth boxes: {n_gt}; correctly found (class right): {(g.status == 'TP').sum()} ({(g.status == 'TP').mean():.1%}); "
          f"found but wrong class: {(g.status == 'CONFUSION').sum()}; missed: {(g.status == 'MISSED').sum()}\n"
          f"- predictions above threshold: {len(pr)}; false positives on background: {(pr.status == 'FP').sum()}\n"
          f"- image level: defect recall {il['any_defect_recall']:.3f}, false-alarm rate on defect-free images {il['any_defect_fpr']:.3f}\n\n"
          "## Write-up prompts (answer with the CSVs/figures in this folder)\n"
          "1. Which classes are missed or confused, and with which other class (confusion_matrix.png)? Is it visual similarity or label noise?\n"
          "2. How does recall change with defect size (recall_by_size.csv)? Does it match Finding 1?\n"
          "3. Grey vs dyed: any gap in recall or false alarms (recall_by_colour.csv)? Does it fit the colour-shortcut result?\n"
          "4. Does crowding (many defects per image) hurt recall (recall_by_crowding.csv)?\n"
          "5. Look at worst_fn.png / worst_fp.png: label noise, duplicate-looking annotations, unusual images? These are your data-quality findings.\n"
          "Do NOT retune on test results; describe errors, do not fix them on the test set.\n")
    (out / "summary.md").write_text(md)
    print(md)
    print(cm.to_string())
    print("outputs ->", out)


if __name__ == "__main__":
    main()
