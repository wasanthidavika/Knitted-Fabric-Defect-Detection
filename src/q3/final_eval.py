#!/usr/bin/env python3
"""
FINAL evaluation on the held-out TEST split. Run ONCE, after every model, threshold and configuration is frozen.

Evaluates in one go, with the shared metric code: trivial benchmark, classical baseline, initial ANN, final ANN
(one or several seeds -> mean +/- std). Writes outputs/q3/final_comparison.{csv,md}, final_eval.json and a lock file
that blocks a second run (use --force only if you must, and say so in the report).

Usage (example):
  python src/final_eval.py --confirm \
     --baseline outputs/q3/baseline/best.joblib \
     --yolo-initial outputs/q3/E00/weights/best.pt --initial-id E00 \
     --yolo-final outputs/q3/F42/weights/best.pt,outputs/q3/F43/weights/best.pt,outputs/q3/F44/weights/best.pt --final-ids F42,F43,F44
Use --skip-yolo only for smoke tests without ultralytics.
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config
from evaluate_detections import evaluate_split
from metrics import trivial_preds
from q3_utils import git_commit, hardware_string, load_gt, log_path, now, q3_dirs, resolve_conf_thr


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def logged_minutes(cfg, ids):
    p = log_path(cfg)
    if not p.exists():
        return [np.nan] * len(ids)
    df = pd.read_csv(p, dtype=str, keep_default_na=False).set_index("id")
    out = []
    for i in ids:
        try:
            out.append(float(df.loc[i, "train_time_min"]))
        except Exception:
            out.append(np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--confirm", action="store_true", help="I confirm everything is frozen and this is the single test evaluation")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--yolo-initial", default=None)
    ap.add_argument("--initial-id", default="E00")
    ap.add_argument("--yolo-final", default=None, help="comma-separated weights (one per seed)")
    ap.add_argument("--final-ids", default=None, help="comma-separated run ids matching --yolo-final")
    ap.add_argument("--skip-yolo", action="store_true")
    ap.add_argument("--conf-thr", type=float, default=0.25, help="fallback image-level threshold (NOT validation-chosen)")
    ap.add_argument("--thr-baseline", default=None, help="threshold JSON from select_threshold.py (val) for the baseline")
    ap.add_argument("--thr-initial", default=None, help="threshold JSON (val) for the initial ANN")
    ap.add_argument("--thr-final", default=None, help="threshold JSON (val) for the final ANN (one threshold for all seeds)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    D = q3_dirs(cfg)
    if not a.confirm:
        raise SystemExit("Add --confirm to acknowledge this is the single, final test evaluation.")
    if D["lock"].exists() and not a.force:
        raise SystemExit(f"{D['lock']} exists: the test set was already evaluated. Re-running requires --force and must be disclosed.")
    D["preds"].mkdir(parents=True, exist_ok=True)
    gt_tr, _ = load_gt(cfg, "train")
    gt_te, imgs_te = load_gt(cfg, "test")
    names = cfg["class_names"]
    results, lock = [], {"time": now(), "git_commit": git_commit(), "hardware": hardware_string(), "weights_md5": {}}

    def add(label, kind, preds, lat, minutes, extra="", thr_json=None):
        preds.to_csv(D["preds"] / f"{label}_test.csv", index=False)
        thr = resolve_conf_thr(a.conf_thr, thr_json)
        if thr_json is None and kind != "trivial benchmark":
            extra = (extra + "; " if extra else "") + f"image-level threshold {thr} is the fallback, not validation-chosen"
        ev = evaluate_split(cfg, preds, "test", thr, allow_test=True)
        results.append({"model": label, "kind": kind, "mAP50": ev["map50"], "mAP50_95": ev["map50_95"],
                        "train_time_min": minutes, "latency_ms_median": lat.get("median_ms"), "hardware": lat.get("hardware", ""),
                        "image_conf_thr": thr, "image_macro_F1": ev["image_level"]["macro_f1"], "image_defect_recall": ev["image_level"]["any_defect_recall"],
                        "notes": extra, **{f"AP50_{k}": v for k, v in ev["ap50_per_class"].items()}})
        print(f"{label}: mAP50={ev['map50']:.4f} mAP50-95={ev['map50_95']:.4f}")

    top = int(gt_tr["cls"].value_counts().idxmax())
    add("trivial_whole_image_" + names[top].replace(" ", ""), "trivial benchmark",
        trivial_preds(imgs_te["image_id"], top), {"median_ms": 0.0, "hardware": "n/a"}, 0.0, "whole-image box of the most frequent TRAIN class, conf 1")

    if a.baseline:
        from baseline_predict import predict_split as svm_predict
        t = json.load(open(D["baseline"] / "timings.json")) if (D["baseline"] / "timings.json").exists() else {}
        minutes = (t.get("stage1_fit_s", 0) + t.get("cache_train_s", 0) + t.get("best_svm_fit_s", 0)) / 60
        preds, lat = svm_predict(cfg, a.baseline, "test", allow_test=True)
        lock["weights_md5"]["baseline"] = md5(a.baseline)
        add("classical_SVM_baseline", "classical baseline", preds, lat, round(minutes, 2), "train time = stage-1 + train features + best SVM fit (grid search excluded)", a.thr_baseline)

    if not a.skip_yolo:
        from yolo_predict import predict_split as yolo_predict
        if a.yolo_initial:
            preds, lat = yolo_predict(cfg, a.yolo_initial, "test", allow_test=True)
            lock["weights_md5"]["yolo_initial"] = md5(a.yolo_initial)
            add("initial_ANN_" + a.initial_id, "initial ANN", preds, lat, logged_minutes(cfg, [a.initial_id])[0], "screening run, 30 epochs", a.thr_initial)
        if a.yolo_final:
            ws = a.yolo_final.split(",")
            ids = a.final_ids.split(",") if a.final_ids else [f"seed{i}" for i in range(len(ws))]
            mins = logged_minutes(cfg, ids)
            for w, i, m in zip(ws, ids, mins):
                preds, lat = yolo_predict(cfg, w, "test", allow_test=True)
                lock["weights_md5"][f"yolo_final_{i}"] = md5(w)
                add(f"final_ANN_{i}", "final ANN", preds, lat, m, "", a.thr_final)

    df = pd.DataFrame(results)
    fin = df[df.kind == "final ANN"]
    if len(fin) > 1:
        agg = {"model": f"final_ANN_mean_over_{len(fin)}_seeds", "kind": "final ANN (mean)"}
        for c in ("mAP50", "mAP50_95", "train_time_min", "latency_ms_median", "image_macro_F1", "image_defect_recall"):
            agg[c] = float(fin[c].mean())
        agg["notes"] = f"std mAP50 = {fin.mAP50.std():.4f}, std mAP50-95 = {fin.mAP50_95.std():.4f}"
        df = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)
    df.to_csv(D["root"] / "final_comparison.csv", index=False)
    cols = ["model", "kind", "mAP50", "mAP50_95", "train_time_min", "latency_ms_median", "hardware", "notes"]
    try:
        table = df[cols].round(4).to_markdown(index=False)
    except ImportError:                      # tabulate not installed
        table = df[cols].round(4).to_string(index=False)
    (D["root"] / "final_comparison.md").write_text(table)
    json.dump(results, open(D["root"] / "final_eval.json", "w"), indent=2, default=str)
    lock["args"] = vars(a)
    json.dump(lock, open(D["lock"], "w"), indent=2, default=str)
    print("\n" + df[cols].round(4).to_string(index=False))
    print(f"\nwrote {D['root'] / 'final_comparison.csv'} and lock file {D['lock']}")


if __name__ == "__main__":
    main()
