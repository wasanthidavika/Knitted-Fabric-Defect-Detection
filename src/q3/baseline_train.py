#!/usr/bin/env python3
"""
Train and tune the classical baseline (Q3.a). Resumable, stage by stage, all caches on disk.

  python src/baseline_train.py --stage fit_stage1   # anchors (k-means on train boxes) + stage-1 objectness model
  python src/baseline_train.py --stage cache --split train
  python src/baseline_train.py --stage cache --split val
  python src/baseline_train.py --stage grid         # stage-2 RBF-SVM grid on the train cache, scored on VAL detection mAP@0.5
  (or --stage all)

Search space (state it in the report):  C in {1, 10, 100} x gamma in {0.001, 0.01, 0.1}  = 9 configurations
Fixed (reported, not tuned): K=300 windows/image, stage-1 NMS IoU 0.5, ridge alpha 1.0, 5 anchors, negatives:positives 1:1,
at most 3000 positive windows per class for the SVM (RBF-SVM training is quadratic in the sample count).
Selection uses VALIDATION only. Test is never touched here.
Every configuration is appended to logs/experiments.csv (family "svm").
"""
import argparse, json, sys, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_lib import extract_candidates, fit_anchors, load_gray, stage1_samples, windows_to_detections
from common import load_config, set_seed
from evaluate_detections import evaluate_split
from metrics import evaluate_detections
from q3_utils import (append_log, git_commit, load_gt, log_path, logged_ids, now, q3_dirs,
                     next_attempt, hardware_string)


def record_run(cfg, experiment_id, family, group, hypothesis, what_changed,
               hyperparameters, started, execution_status, scientific_outcome="pending_review",
               metrics=None, auto_flags="", run_dir="", what_was_learned=""):
    """Record one actual execution. Repeated IDs receive increasing attempt numbers."""
    elapsed_min = (time.time() - started) / 60.0
    row = {
        "id": experiment_id,
        "attempt": next_attempt(log_path(cfg), experiment_id),
        "date": now(),
        "family": family,
        "group": group,
        "hypothesis": hypothesis,
        "what_changed": what_changed,
        "hyperparameters": hyperparameters,
        "seed": cfg.get("seed", ""),
        "train_time_min": round(elapsed_min, 2),
        "outcome": execution_status,
        "execution_status": execution_status,
        "scientific_outcome": scientific_outcome,
        "auto_flags": auto_flags,
        "what_was_learned": what_was_learned,
        "run_dir": str(run_dir),
        "git_commit": git_commit(),
        "hardware": hardware_string(),
    }
    if metrics:
        row.update(metrics)
    append_log(log_path(cfg), row)


def run_stage_with_log(cfg, experiment_id, group, hypothesis, what_changed,
                       hyperparameters, run_dir, function):
    """Run a baseline stage and record both successful and failed executions."""
    started = time.time()
    try:
        result = function()
        record_run(
            cfg, experiment_id, "classical_baseline", group, hypothesis,
            what_changed, hyperparameters, started, "completed",
            run_dir=run_dir,
            what_was_learned="Stage completed. Review its saved output and record the scientific finding.",
        )
        return result
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        record_run(
            cfg, experiment_id, "classical_baseline", group, hypothesis,
            what_changed, hyperparameters, started, "failed",
            scientific_outcome="not_evaluated",
            auto_flags=f"{type(exc).__name__}: {exc}",
            run_dir=run_dir,
            what_was_learned="Execution failed. Diagnose the error before rerunning.",
        )
        raise


def img_path(cfg, split, iid):
    return Path(cfg["resized_dir"]) / "images" / split / f"{iid}.jpg"


def gt_by_image(gt):
    out = {}
    for iid, g in gt.groupby("image_id"):
        px = np.stack([g.x1 * g.rW, g.y1 * g.rH, g.x2 * g.rW, g.y2 * g.rH], 1).astype(float)
        out[iid] = (px, g["cls"].values)
    return out


def _stage1_job(path, anchors, gt_px, seed, n_pos, n_neg):
    import cv2
    cv2.setNumThreads(1)
    return stage1_samples(load_gray(path), anchors, gt_px, np.random.default_rng(seed), n_pos, n_neg)


def _cache_job(path, out, anchors, stage1, K, nms_iou, gt_px, gt_cls):
    import cv2
    cv2.setNumThreads(1)
    if Path(out).exists():
        return 0
    c = extract_candidates(load_gray(path), anchors, stage1, K, nms_iou, gt_px, gt_cls)
    np.savez_compressed(out, **c)
    return 1


def stage_fit_stage1(cfg, a, D):
    gt, imgs = load_gt(cfg, "train")
    G = gt_by_image(gt)
    ids = sorted(imgs["image_id"])
    rng = np.random.default_rng(cfg["seed"])
    if a.stage1_images and a.stage1_images < len(ids):
        ids = list(rng.choice(ids, a.stage1_images, replace=False))
    wh = np.stack([(gt.x2 - gt.x1) * gt.rW, (gt.y2 - gt.y1) * gt.rH], 1)
    anchors = fit_anchors(wh, a.anchors, cfg["seed"])
    print("anchors (w,h px at resized scale):", anchors.tolist())
    t0 = time.time()
    res = Parallel(n_jobs=a.workers)(delayed(_stage1_job)(img_path(cfg, "train", i), anchors, G.get(i, (np.zeros((0, 4)), None))[0],
                                                          cfg["seed"] + k, 30, 150) for k, i in enumerate(ids))
    X, y = np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])
    perm = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    model.fit(X[perm[:cut]], y[perm[:cut]])
    auc = roc_auc_score(y[perm[cut:]], model.decision_function(X[perm[cut:]]))
    model.fit(X, y)
    secs = time.time() - t0
    D["baseline"].mkdir(parents=True, exist_ok=True)
    joblib.dump({"anchors": anchors, "stage1": model, "K": a.K, "nms_iou": a.nms_iou, "stage1_images": len(ids)}, D["baseline"] / "stage1.joblib")
    tj = D["baseline"] / "timings.json"
    t = json.load(open(tj)) if tj.exists() else {}
    t["stage1_fit_s"] = round(secs, 1)
    json.dump(t, open(tj, "w"), indent=2)
    print(f"stage-1: {len(ids)} images, {len(y)} windows ({int(y.sum())} positive), held-out AUC {auc:.3f}, {secs:.0f}s")


def cache_dir(D, split):
    return D["baseline"] / "cache" / split


def stage_cache(cfg, a, D):
    if a.split not in ("train", "val"):
        raise SystemExit("cache is built for train/val here; test features are computed inside baseline_predict at final evaluation")
    s1 = joblib.load(D["baseline"] / "stage1.joblib")
    gt, imgs = load_gt(cfg, a.split)
    G = gt_by_image(gt) if a.split == "train" else {}
    ids = sorted(imgs["image_id"])[: a.limit] if a.limit else sorted(imgs["image_id"])
    out_dir = cache_dir(D, a.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"anchors": s1["anchors"].tolist(), "K": s1["K"], "nms_iou": s1["nms_iou"]}
    mp = out_dir / "cache_meta.json"
    if mp.exists() and json.load(open(mp)) != meta:
        raise SystemExit(f"{mp} was built with different anchors/K: delete {out_dir} and rebuild")
    json.dump(meta, open(mp, "w"))
    t0 = time.time()
    n = Parallel(n_jobs=a.workers)(delayed(_cache_job)(img_path(cfg, a.split, i), out_dir / f"{i}.npz", s1["anchors"], s1["stage1"],
                                                       s1["K"], s1["nms_iou"], *( [G[i][0], G[i][1]] if (a.split == "train" and i in G) else
                                                       ([np.zeros((0, 4)), np.zeros(0)] if a.split == "train" else [None, None]))) for i in ids)
    secs = time.time() - t0
    tj = D["baseline"] / "timings.json"
    t = json.load(open(tj)) if tj.exists() else {}
    t[f"cache_{a.split}_s"] = round(t.get(f"cache_{a.split}_s", 0) + secs, 1)
    json.dump(t, open(tj, "w"), indent=2)
    print(f"cached {sum(n)} new / {len(ids)} images for {a.split} in {secs:.0f}s -> {out_dir}")


def load_cache(D, split, ids):
    out = {}
    for i in ids:
        p = cache_dir(D, split) / f"{i}.npz"
        if p.exists():
            out[i] = dict(np.load(p))
    return out


def predict_cached(bundle, cache, regress=True):
    dfs = [windows_to_detections(bundle, iid, c["boxes"], c["X"], c["hw"], regress=regress) for iid, c in cache.items()]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def stage_grid(cfg, a, D):
    s1 = joblib.load(D["baseline"] / "stage1.joblib")
    n_cls = len(cfg["class_names"])
    bg = n_cls
    gt_tr, imgs_tr = load_gt(cfg, "train")
    cache = load_cache(D, "train", imgs_tr["image_id"])
    if not cache:
        raise SystemExit("train cache is empty: run --stage cache --split train first")
    X = np.concatenate([c["X"] for c in cache.values()])
    iou = np.concatenate([c["iou_best"] for c in cache.values()])
    cl = np.concatenate([c["cls_best"] for c in cache.values()]).astype(int)
    T = np.concatenate([c["targets"] for c in cache.values()])
    rng = np.random.default_rng(cfg["seed"])
    pos_idx = []
    for c in range(n_cls):
        idx = np.where((iou >= 0.5) & (cl == c))[0]
        pos_idx.append(rng.choice(idx, min(a.max_pos_per_class, len(idx)), replace=False) if len(idx) else idx)
    pos = np.concatenate(pos_idx).astype(int)
    neg_all = np.where(iou < 0.3)[0]
    neg = rng.choice(neg_all, min(len(neg_all), int(a.bg_ratio * len(pos))), replace=False)
    sel = np.r_[pos, neg]
    y = np.r_[cl[pos], np.full(len(neg), bg)]
    print(f"stage-2 training set: {len(pos)} positive windows {np.bincount(cl[pos], minlength=n_cls).tolist()} + {len(neg)} background")
    scaler = StandardScaler().fit(X[sel])
    Xs = scaler.transform(X[sel])
    ridge = Ridge(alpha=a.ridge_alpha).fit(scaler.transform(X[pos]), T[pos])

    gt_v, imgs_v = load_gt(cfg, "val")
    cache_v = load_cache(D, "val", imgs_v["image_id"])
    if not cache_v:
        raise SystemExit("val cache is empty: run --stage cache --split val first")
    # upper bound on recall of the cascade: GT boxes that have >=1 candidate window with IoU>=0.5
    from metrics import iou_matrix
    cov, tot = 0, 0
    for iid, g in gt_v.groupby("image_id"):
        if iid in cache_v:
            px = np.stack([g.x1 * g.rW, g.y1 * g.rH, g.x2 * g.rW, g.y2 * g.rH], 1)
            cov += int((iou_matrix(px, cache_v[iid]["boxes"].astype(float)).max(1) >= 0.5).sum())
            tot += len(g)
    stage1_recall = cov / max(tot, 1)
    print(f"cascade recall upper bound on val (GT boxes with a candidate window IoU>=0.5): {stage1_recall:.3f}")

    Cs, Gs = ([1, 10], [0.01, 0.1]) if a.quick else ([1, 10, 100], [0.001, 0.01, 0.1])
    rows, best = [], None
    k = 0
    for C in Cs:
        for gam in Gs:
            k += 1
            cid = f"S{k:02d}"
            t0 = time.time()
            svm = SVC(kernel="rbf", C=C, gamma=gam, cache_size=2000, decision_function_shape="ovr").fit(Xs, y)
            fit_s = time.time() - t0
            bundle = {"anchors": s1["anchors"], "stage1": s1["stage1"], "K": s1["K"], "nms_iou": s1["nms_iou"], "scaler": scaler,
                      "svm": svm, "ridge": ridge, "bg_label": bg, "n_classes": n_cls, "feature_names": "see baseline_lib.FEATURE_NAMES"}
            t1 = time.time()
            preds = predict_cached(bundle, cache_v)
            res = evaluate_detections(preds, gt_v, sorted(cfg["class_names"]))
            pred_s = time.time() - t1
            rows.append({"id": cid, "C": C, "gamma": gam, "val_mAP50": round(res["map50"], 4), "val_mAP50_95": round(res["map50_95"], 4),
                         "fit_s": round(fit_s, 1), "val_predict_s": round(pred_s, 1), "n_support": int(svm.n_support_.sum())})
            print(rows[-1])
            append_log(log_path(cfg), {
                "id": cid,
                "attempt": next_attempt(log_path(cfg), cid),
                "date": now(),
                "family": "svm",
                "group": "baseline_grid",
                "hypothesis": "RBF-SVM on Gabor+GLCM+LBP window features; C/gamma grid chosen on validation mAP50",
                "what_changed": f"C={C}, gamma={gam}",
                "seed": cfg["seed"],
                "hyperparameters": (
                    f"C={C} gamma={gam} K={s1['K']} "
                    f"anchors={s1['anchors'].tolist()} ridge_alpha={a.ridge_alpha} "
                    f"train_windows={len(sel)}"
                ),
                "val_mAP50": rows[-1]["val_mAP50"],
                "val_mAP50_95": rows[-1]["val_mAP50_95"],
                "train_time_min": round(fit_s / 60, 2),
                "outcome": "completed",
                "execution_status": "completed",
                "scientific_outcome": "pending_review",
                "auto_flags": "",
                "what_was_learned": "Validation result recorded. Add interpretation after comparing configurations.",
                "run_dir": str(D["baseline"]),
                "git_commit": git_commit(),
                "hardware": hardware_string(),
            })
            if best is None or res["map50"] > best[0]:
                best = (res["map50"], bundle, cid, fit_s)
    grid = pd.DataFrame(rows)
    # post-processing setting: per-class NMS IoU, chosen on VAL for the best SVM (3 extra evaluations, no retraining)
    nms_rows, best_nms = [], (-1.0, 0.5)
    for thr in (0.3, 0.4, 0.5):
        best[1]["per_class_nms"] = thr
        r_ = evaluate_detections(predict_cached(best[1], cache_v), gt_v, sorted(cfg["class_names"]))
        nms_rows.append({"id": f"N{int(thr * 100)}", "per_class_nms": thr, "val_mAP50": round(r_["map50"], 4), "val_mAP50_95": round(r_["map50_95"], 4)})
        print(nms_rows[-1])
        append_log(log_path(cfg), {
            "id": f"N{int(thr * 100)}",
            "attempt": next_attempt(log_path(cfg), f"N{int(thr * 100)}"),
            "date": now(),
            "family": "svm",
            "group": "baseline_postproc",
            "hypothesis": "Lower NMS IoU removes stacked duplicate boxes around one defect and raises precision",
            "what_changed": f"per-class NMS IoU={thr} on best SVM {best[2]}",
            "seed": cfg["seed"],
            "hyperparameters": f"svm={best[2]} per_class_nms={thr}",
            "val_mAP50": nms_rows[-1]["val_mAP50"],
            "val_mAP50_95": nms_rows[-1]["val_mAP50_95"],
            "train_time_min": 0,
            "outcome": "completed",
            "execution_status": "completed",
            "scientific_outcome": "pending_review",
            "auto_flags": "",
            "what_was_learned": "Validation result recorded. Compare NMS settings before selecting one.",
            "run_dir": str(D["baseline"]),
            "git_commit": git_commit(),
            "hardware": hardware_string(),
        })
        if r_["map50"] > best_nms[0]:
            best_nms = (r_["map50"], thr)
    best[1]["per_class_nms"] = best_nms[1]
    pd.concat([grid, pd.DataFrame(nms_rows)], ignore_index=True).to_csv(D["baseline"] / "grid_results.csv", index=False)
    joblib.dump(best[1], D["baseline"] / "best.joblib")
    # ablation reported honestly: effect of box regression and of the cascade on the best config
    pred_nr = predict_cached(best[1], cache_v, regress=False)
    nr = evaluate_detections(pred_nr, gt_v, sorted(cfg["class_names"]))
    tj = D["baseline"] / "timings.json"
    t = json.load(open(tj)) if tj.exists() else {}
    t.update({"best_config": best[2], "best_svm_fit_s": round(best[3], 1)})
    json.dump(t, open(tj, "w"), indent=2)
    summ = {"configs_evaluated": len(grid) + len(nms_rows), "svm_configs": len(grid), "nms_settings": len(nms_rows),
            "chosen_per_class_nms": best_nms[1], "best_id": best[2], "best_val_mAP50_svm_only": round(best[0], 4),
            "best_val_mAP50_after_nms_choice": round(best_nms[0], 4),
            "best_val_mAP50_without_box_regression": round(nr["map50"], 4), "cascade_recall_upper_bound_val": round(stage1_recall, 4),
            "search_space": {"C": Cs, "gamma": Gs}, "train_windows": int(len(sel))}
    json.dump(summ, open(D["baseline"] / "baseline_summary.json", "w"), indent=2)
    print("\n", json.dumps(summ, indent=2))
    print(f"best bundle -> {D['baseline'] / 'best.joblib'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stage", required=True, choices=["fit_stage1", "cache", "grid", "all"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=2, help="parallel jobs (Kaggle CPU has few cores)")
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N images when caching")
    ap.add_argument("--stage1-images", type=int, default=400)
    ap.add_argument("--anchors", type=int, default=5)
    ap.add_argument("--K", type=int, default=300)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--max-pos-per-class", type=int, default=3000)
    ap.add_argument("--bg-ratio", type=float, default=1.0)
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--quick", action="store_true", help="2x2 grid (smoke test only)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    D = q3_dirs(cfg)
    D["baseline"].mkdir(parents=True, exist_ok=True)

    if a.stage in ("fit_stage1", "all"):
        params = (
            f"stage1_images={a.stage1_images}; anchors={a.anchors}; "
            f"K={a.K}; nms_iou={a.nms_iou}; seed={cfg['seed']}"
        )
        run_stage_with_log(
            cfg, "BASELINE_STAGE1", "baseline_stage1",
            "Train anchor-based stage-1 objectness model using training images only.",
            f"stage1_images={a.stage1_images}",
            params, D["baseline"],
            lambda: stage_fit_stage1(cfg, a, D),
        )

    if a.stage in ("cache", "all"):
        splits = ["train", "val"] if a.stage == "all" else [a.split]
        for sp in splits:
            if sp not in ("train", "val"):
                raise SystemExit("Cache is restricted to train/val. Test features are generated only during final evaluation.")
            a.split = sp
            params = (
                f"split={sp}; limit={a.limit}; workers={a.workers}; "
                f"K={a.K}; seed={cfg['seed']}"
            )
            run_stage_with_log(
                cfg, f"BASELINE_CACHE_{sp.upper()}", "baseline_feature_cache",
                "Generate reusable candidate-window features for the specified split.",
                f"split={sp}; limit={a.limit}",
                params, cache_dir(D, sp),
                lambda sp=sp: stage_cache(cfg, a, D),
            )

    if a.stage in ("grid", "all"):
        params = (
            f"C={'[1, 10]' if a.quick else '[1, 10, 100]'}; "
            f"gamma={'[0.01, 0.1]' if a.quick else '[0.001, 0.01, 0.1]'}; "
            f"max_pos_per_class={a.max_pos_per_class}; bg_ratio={a.bg_ratio}; "
            f"ridge_alpha={a.ridge_alpha}; seed={cfg['seed']}"
        )
        run_stage_with_log(
            cfg, "BASELINE_GRID_STAGE", "baseline_grid_stage",
            "Compare the planned RBF-SVM configurations using validation detection mAP.",
            "Run SVM grid and validation-selected NMS evaluations.",
            params, D["baseline"],
            lambda: stage_grid(cfg, a, D),
        )


if __name__ == "__main__":
    main()
