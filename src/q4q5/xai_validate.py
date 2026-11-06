#!/usr/bin/env python3
"""
Q5.b validation of the explanations. Three parts, all logged to logs/experiments.csv (family "xai").

1. DELETION SANITY TEST (default): for N correctly detected defects, hide the window the explanation highlights most vs random
   windows of the same size, and compare the drop in the detector's confidence for THAT detection.
   Methods: occlusion (self-consistency: it is built from the same kind of perturbation), eigencam (independent of occlusion:
   the real faithfulness test), and a CONTROL "box_centre" (just highlight the predicted box). An explanation is only
   informative if it beats the control; report honestly if it does not.
2. --cases : case-study figures: correct detections, false positives, class confusions and missed defects (>= 3 incorrect).
3. --randomise : model-randomisation test (YOLO only): EigenCAM must change when the network weights are randomised.

Examples:
  python src/xai_validate.py --backend yolo --weights outputs/q3/F42/weights/best.pt --split val \
         --preds outputs/q3/preds/F42_val.csv --threshold-json outputs/q3/eval/threshold_F42.json --n 30 --cases --randomise
  python src/xai_validate.py --backend mock --split val --n 5 --cases        # test without a trained model
Use val while developing. --split test only after final_eval.py (lock file) and only for illustrating the FINAL model.
"""
import argparse, json, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config
from detectors import build_detector
from error_analysis import match
from metrics import iou_matrix
from q3_utils import append_log, git_commit, load_gt, log_path, now, q3_dirs, resolve_conf_thr
import xai_core as X


def load_img(cfg, split, iid):
    im = cv2.imread(str(Path(cfg["resized_dir"]) / "images" / split / f"{iid}.jpg"))
    return im[..., ::-1].copy()


def centre_heat(shape, box):
    H, W = shape
    cx, cy = (box[0] + box[2]) / 2 * W, (box[1] + box[3]) / 2 * H
    s = max((box[2] - box[0]) * W, (box[3] - box[1]) * H, 8)
    yy, xx = np.mgrid[0:H, 0:W]
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)).astype(np.float32)


def boot_ci(x, n=2000, seed=0):
    r = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = [r.choice(x, len(x)).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarise(method, top, rnd):
    from scipy.stats import wilcoxon
    top, rnd = np.asarray(top, float), np.asarray(rnd, float)
    diff = top - rnd
    try:
        p = float(wilcoxon(diff, alternative="greater").pvalue)
    except ValueError:
        p = None
    lo, hi = boot_ci(diff) if len(diff) > 1 else (float("nan"), float("nan"))
    return {"method": method, "n": int(len(diff)), "mean_drop_top_window": float(top.mean()), "mean_drop_random_windows": float(rnd.mean()),
            "mean_difference": float(diff.mean()), "diff_ci95": [lo, hi], "share_top_greater_than_random": float((diff > 0).mean()),
            "wilcoxon_p_top_greater": p}


def predict_all(detector, cfg, split, imgs, floor, max_images):
    rows = []
    for iid in imgs["image_id"].tolist()[:max_images]:
        d = detector.predict(load_img(cfg, split, iid), floor)
        for r in d.itertuples():
            rows.append((iid, r.cls, r.conf, r.x1, r.y1, r.x2, r.y2))
    return pd.DataFrame(rows, columns=["image_id", "cls", "conf", "x1", "y1", "x2", "y2"])


def case_figure(cfg, split, kind, row, res, gt_img, preds_img, names, path):
    img = row["img"]
    H, W = img.shape[:2]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    base = img.copy()
    for g in gt_img.itertuples():
        cv2.rectangle(base, (int(g.x1 * W), int(g.y1 * H)), (int(g.x2 * W), int(g.y2 * H)), (0, 255, 0), 2)
    for q in preds_img.itertuples():
        cv2.rectangle(base, (int(q.x1 * W), int(q.y1 * H)), (int(q.x2 * W), int(q.y2 * H)), (255, 0, 0), 1)
    b = row["box"]
    cv2.rectangle(base, (int(b[0] * W), int(b[1] * H)), (int(b[2] * W), int(b[3] * H)), (255, 255, 0), 3)
    ax[0].imshow(base)
    ax[0].set_title(f"{kind}: target (yellow) | green = truth, red = predictions", fontsize=9)
    for a_, m, title in ((ax[1], "occlusion", "Occlusion check"), (ax[2], "eigencam", "EigenCAM overview")):
        d = res.get(m, {"available": False, "reason": "EigenCAM needs the YOLO (CNN) backend" if m == "eigencam" else "not run"})
        if d.get("available"):
            a_.imshow(X.overlay(img, d["heat_norm"], box_norm=b))
            sub = f"hide top window: {d['s0']:.2f}->{d['score_top_hidden']:.2f} (random: drop {d['drop_random']:.2f})" if "drop_top" in d else ""
            a_.set_title(f"{title}\n{sub}", fontsize=9)
        else:
            a_.text(0.5, 0.5, d.get("reason", "n/a"), ha="center", va="center", wrap=True, fontsize=9)
            a_.set_title(title, fontsize=9)
    for a_ in ax:
        a_.axis("off")
    fig.suptitle(f"{row['title']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--backend", default="yolo", choices=["yolo", "svm", "mock"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--preds", default=None, help="predictions CSV (yolo_predict.py); omitted = run the detector now")
    ap.add_argument("--max-images", type=int, default=10**6)
    ap.add_argument("--threshold-json", default=None)
    ap.add_argument("--n", type=int, default=30, help="correctly detected defects for the deletion test")
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--layer", type=int, default=15, help="EigenCAM layer: 15 / 18 / 21 = stride 8 / 16 / 32 (YOLOv8)")
    ap.add_argument("--max-patches", type=int, default=36)
    ap.add_argument("--fill", default="inpaint", choices=["inpaint", "blur", "mean"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--cases", action="store_true")
    ap.add_argument("--randomise", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    names = cfg["class_names"]
    D = q3_dirs(cfg)
    if a.split == "test" and not D["lock"].exists():
        raise SystemExit("Test split is only allowed after final_eval.py (lock file missing).")
    out = Path(cfg["outputs_dir"]) / "q5"
    (out / "cases").mkdir(parents=True, exist_ok=True)
    detector = build_detector(a.backend, names, a.weights, a.bundle, a.device)
    gt, imgs = load_gt(cfg, a.split)
    preds = pd.read_csv(a.preds) if a.preds else predict_all(detector, cfg, a.split, imgs, 0.05, a.max_images)
    thr = resolve_conf_thr(None, a.threshold_json)
    pr, g = match(preds, gt, thr)
    rng = np.random.default_rng(a.seed)
    methods = ["occlusion"] + (["eigencam"] if detector.supports_eigencam else [])
    print(f"backend={a.backend} thr={thr} detections>=thr={len(pr)} TP={int((pr.status=='TP').sum())} FP={int((pr.status=='FP').sum())} "
          f"CONF={int((pr.status=='CONFUSION').sum())} FN={int((g.status=='MISSED').sum())}")

    # ---------------------------------------------------------------- 1. deletion test
    tp = pr[pr.status == "TP"]
    if len(tp) == 0:
        print("No correctly detected defects at this threshold: cannot run the deletion test.")
    sample = tp.sample(min(a.n, len(tp)), random_state=a.seed) if len(tp) else tp
    rows, cache = [], {}
    for k, r in enumerate(sample.itertuples()):
        img = cache.setdefault(r.image_id, load_img(cfg, a.split, r.image_id))
        box = np.array([r.x1, r.y1, r.x2, r.y2], float)
        res = X.explain_detection(detector, img, box, int(r.cls), methods, a.layer, rng, a.max_patches, a.n_random, a.fill)
        p, roi = X.window_geometry(img.shape[:2], box)
        ctrl = X.deletion_test(detector, img, box, int(r.cls), centre_heat(img.shape[:2], box), p, roi, rng, a.n_random, a.fill, s0=res["s0"])
        row = {"image_id": r.image_id, "cls": int(r.cls), "conf": float(r.conf), "s0": res["s0"], "patch_px": p,
               "box_centre_drop_top": ctrl["drop_top"], "box_centre_drop_random": ctrl["drop_random"]}
        for m in methods:
            d = res.get(m, {})
            if d.get("available") and "drop_top" in d:
                row[f"{m}_drop_top"], row[f"{m}_drop_random"] = d["drop_top"], d["drop_random"]
        rows.append(row)
        if (k + 1) % 5 == 0:
            print(f"  deletion test {k + 1}/{len(sample)}")
    det = pd.DataFrame(rows)
    det.to_csv(out / f"deletion_test_{a.tag}.csv", index=False)
    summ = []
    for m in methods + ["box_centre"]:
        t, r_ = f"{m}_drop_top", f"{m}_drop_random"
        if len(det) and t in det.columns:
            s = summarise(m, det[t].dropna(), det.loc[det[t].notna(), r_])
            summ.append(s)
            append_log(log_path(cfg), {"id": f"XAI_{a.tag}_{m}", "date": now(), "family": "xai", "group": "xai_deletion_test",
                       "hypothesis": "Hiding the window an explanation highlights lowers the detector confidence more than hiding a random window of the same size",
                       "what_changed": f"{m} on {s['n']} TP detections ({a.split}), backend={a.backend}, layer={a.layer}, fill={a.fill}",
                       "hyperparameters": f"n_random={a.n_random} max_patches={a.max_patches} patch=0.8*box seed={a.seed}", "seed": a.seed,
                       "outcome": "completed", "auto_flags": "", "run_dir": str(out), "git_commit": git_commit(),
                       "what_was_learned": f"[AUTO] top-window drop {s['mean_drop_top_window']:.3f} vs random {s['mean_drop_random_windows']:.3f}; "
                                           f"top>random in {s['share_top_greater_than_random']:.0%}; Wilcoxon p={s['wilcoxon_p_top_greater']}. WRITE YOUR INTERPRETATION."})
    json.dump(summ, open(out / f"deletion_summary_{a.tag}.json", "w"), indent=2)
    for s in summ:
        print(json.dumps(s))
    if len(det):
        fig, ax = plt.subplots(figsize=(7, 4))
        data, labels = [], []
        for s in summ:
            m = s["method"]
            data += [det[f"{m}_drop_top"].dropna(), det[f"{m}_drop_random"].dropna()]
            labels += [f"{m}\ntop window", f"{m}\nrandom"]
        ax.boxplot(data)
        ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=8)
        ax.set(ylabel="drop in detector confidence when hidden", title="Deletion sanity test (higher for 'top window' = more faithful)")
        fig.tight_layout()
        fig.savefig(out / f"deletion_test_{a.tag}.png", dpi=130)
        plt.close(fig)

    # ---------------------------------------------------------------- 2. case studies
    if a.cases:
        picks = []
        for r in tp.sample(min(2, len(tp)), random_state=a.seed).itertuples():
            picks.append(("correct", r._asdict(), None))
        for r in pr[pr.status == "FP"].sort_values("conf", ascending=False).head(3).itertuples():
            picks.append(("false_positive", r._asdict(), None))
        for r in pr[pr.status == "CONFUSION"].sort_values("conf", ascending=False).head(2).itertuples():
            picks.append(("class_confusion", r._asdict(), None))
        for r in g[g.status == "MISSED"].sample(min(3, int((g.status == "MISSED").sum())), random_state=a.seed).itertuples():
            picks.append(("missed", r._asdict(), None))
        crow = []
        for k, (kind, r, _) in enumerate(picks):
            iid = r["image_id"]
            img = cache.get(iid) if iid in cache else load_img(cfg, a.split, iid)
            box = np.array([r["x1"], r["y1"], r["x2"], r["y2"]], float)
            cls = int(r["cls"])
            res = X.explain_detection(detector, img, box, cls, methods, a.layer, rng, a.max_patches, a.n_random, a.fill)
            gcls = int(r["gt_cls"]) if kind in ("correct", "class_confusion") and r.get("gt_cls", -1) >= 0 else (cls if kind == "missed" else -1)
            conf = float(r["conf"]) if "conf" in r and kind != "missed" else res["s0"]
            title = (f"{kind.replace('_', ' ')} | image {iid[:8]} | predicted: {names.get(cls, cls)} (score {conf:.2f})"
                     + (f" | truth: {names.get(gcls, gcls)}" if gcls >= 0 else " | no matching true defect"))
            case_figure(cfg, a.split, kind, {"img": img, "box": box, "title": title}, res, gt[gt.image_id == iid], pr[(pr.image_id == iid)], names,
                        out / "cases" / f"{k:02d}_{kind}_{iid[:8]}.png")
            crow.append({"case": k, "kind": kind, "image_id": iid, "pred_or_gt_class": names.get(cls, cls), "truth_class": names.get(gcls, gcls) if gcls >= 0 else "",
                         "score": conf, "occlusion_drop_top": res.get("occlusion", {}).get("drop_top"),
                         "occlusion_drop_random": res.get("occlusion", {}).get("drop_random"),
                         "eigencam_drop_top": res.get("eigencam", {}).get("drop_top"), "eigencam_drop_random": res.get("eigencam", {}).get("drop_random")})
            print("case", k, kind, iid[:8])
        pd.DataFrame(crow).to_csv(out / "cases" / "cases.csv", index=False)
        n_wrong = sum(c["kind"] != "correct" for c in crow)
        print(f"{n_wrong} incorrect-prediction cases written to {out / 'cases'} (the brief's checklist asks for at least 3: analyse each one in the report)")

    # ---------------------------------------------------------------- 3. model randomisation (YOLO only)
    if a.randomise:
        if a.backend != "yolo":
            print("--randomise needs the YOLO backend; skipped")
        else:
            from detectors import YoloDetector
            from scipy.stats import spearmanr
            pick = sample.head(min(15, len(sample)))
            ims = [cache.setdefault(i, load_img(cfg, a.split, i)) for i in pick["image_id"]]
            base = [cv2.resize(X.eigencam_heatmap(detector, im, a.layer), (64, 48)).ravel() for im in ims]
            res_rows = []
            for level, start in (("last_block_only", a.layer), ("neck_and_last_block", 10), ("whole_network_to_layer", 0)):
                d2 = YoloDetector(detector.weights, names)
                net = d2.yolo.model.model
                for i in range(start, a.layer + 1):
                    for mod in net[i].modules():
                        if hasattr(mod, "reset_parameters"):
                            mod.reset_parameters()
                rho = [spearmanr(b, cv2.resize(X.eigencam_heatmap(d2, im, a.layer), (64, 48)).ravel())[0] for b, im in zip(base, ims)]
                res_rows.append({"randomised": level, "mean_spearman_with_trained_map": float(np.nanmean(rho)), "n_images": len(ims)})
                print(res_rows[-1])
            pd.DataFrame(res_rows).to_csv(out / f"randomisation_{a.tag}.csv", index=False)
            append_log(log_path(cfg), {"id": f"XAI_{a.tag}_randomisation", "date": now(), "family": "xai", "group": "xai_randomisation",
                       "hypothesis": "EigenCAM maps from a weight-randomised network differ from the trained model's maps (low rank correlation)",
                       "what_changed": f"EigenCAM layer {a.layer}; randomised: " + ", ".join(r["randomised"] for r in res_rows), "seed": a.seed,
                       "outcome": "completed", "run_dir": str(out), "git_commit": git_commit(),
                       "what_was_learned": "[AUTO] " + "; ".join(f"{r['randomised']}: rho={r['mean_spearman_with_trained_map']:.2f}" for r in res_rows) + ". WRITE YOUR INTERPRETATION."})
    print(f"\noutputs -> {out}")


if __name__ == "__main__":
    main()
