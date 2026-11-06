#!/usr/bin/env python3
"""
FINAL image-level comparison on the held-out TEST split (Q6.a + Q6.b). Run ONCE, after everything is frozen
(ideally in the same session as src/final_eval.py, so the test set is evaluated at one moment).

Compares on the SAME test images with the SAME metric (image-level macro-F1, per-class thresholds tuned on VALIDATION):
final YOLO, classical SVM, fine-tuned ViT, CLIP zero-shot. Adds bootstrap 95% CIs, paired bootstrap differences, defective-vs-clean
operating points chosen on validation, latency, training data/time, and the qualitative cost/privacy/dependence columns
(configs/q6_models.yaml). Writes outputs/q6/q6_comparison.{csv,md}, q6_final.json and a lock file.

Example:
  python src/q6_final_eval.py --confirm \
    --yolo-val outputs/q3/preds/F42_val.csv --yolo-test outputs/q3/preds/final_ANN_F42_test.csv --yolo-label final_ANN_F42 \
    --svm-val outputs/q3/preds/SVM_BEST_val.csv --svm-test outputs/q3/preds/classical_SVM_baseline_test.csv \
    --vit-weights outputs/q6/vit/V00/best.pt --vit-name V00 --clip
(the *_test.csv files are written by src/final_eval.py). Set Q6_FAKE=1 for pipeline tests. --skip-vit when torch/timm are unavailable.
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import image_level as L
from common import load_config
from q3_utils import git_commit, hardware_string, load_gt, log_path, now, q3_dirs


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def table_md(df):
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def logged_value(cfg, run_id, col):
    p = log_path(cfg)
    if not p.exists():
        return None
    d = pd.read_csv(p, dtype=str, keep_default_na=False).set_index("id")
    try:
        v = d.loc[run_id, col]
        return float(v) if v != "" else None
    except Exception:
        return None


def comparison_row(cfg, label):
    p = Path(cfg["outputs_dir"]) / "q3" / "final_comparison.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p).set_index("model")
    return d.loc[label].to_dict() if label in d.index else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--yolo-val"); ap.add_argument("--yolo-test"); ap.add_argument("--yolo-label", default="final_ANN_F42")
    ap.add_argument("--svm-val"); ap.add_argument("--svm-test"); ap.add_argument("--svm-label", default="classical_SVM_baseline")
    ap.add_argument("--vit-weights"); ap.add_argument("--vit-name", default="V00"); ap.add_argument("--skip-vit", action="store_true")
    ap.add_argument("--clip", action="store_true"); ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--models-yaml", default=str(Path(__file__).resolve().parents[1] / "configs" / "q6_models.yaml"))
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--reference", default=None, help="model key for paired comparisons (default: first of yolo, svm)")
    ap.add_argument("--gpu-usd-per-hour", type=float, default=None, help="OPTIONAL notional compute price to turn latency into a cost estimate (your assumption)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    D = q3_dirs(cfg)
    out = Path(cfg["outputs_dir"]) / "q6"
    lock = out / "TEST_EVALUATED_Q6.lock"
    if not a.confirm:
        raise SystemExit("Add --confirm: this is the single, final test evaluation of the Q6 models.")
    if lock.exists() and not a.force:
        raise SystemExit(f"{lock} exists: the Q6 models were already evaluated on test. --force needs disclosure in the report.")
    if not D["lock"].exists():
        print("NOTE: src/final_eval.py has not been run yet. Evaluate everything at one moment: freeze, then run final_eval.py and this script back to back.")
    out.mkdir(parents=True, exist_ok=True)
    meta = yaml.safe_load(open(a.models_yaml))["models"]
    cls_ids = sorted(cfg["class_names"])
    names = cfg["class_names"]
    gt_v, im_v = load_gt(cfg, "val")
    gt_t, im_t = load_gt(cfg, "test")
    ids_v, ids_t = sorted(im_v["image_id"]), sorted(im_t["image_id"])
    y_v, y_t = L.multi_hot(gt_v, ids_v, cls_ids), L.multi_hot(gt_t, ids_t, cls_ids)

    models = {}      # key -> dict(scores_val, scores_test, latency_ms, hardware, train_min, params)
    lock_info = {"time": now(), "git_commit": git_commit(), "weights_md5": {}}

    def from_dets(key, vcsv, tcsv, label):
        if not (vcsv and tcsv):
            return
        row = comparison_row(cfg, label)
        models[key] = {"sv": L.scores_from_detections(pd.read_csv(vcsv), ids_v, cls_ids), "st": L.scores_from_detections(pd.read_csv(tcsv), ids_t, cls_ids),
                       "lat": row.get("latency_ms_median"), "hw": row.get("hardware", ""), "train_min": row.get("train_time_min"), "params": None,
                       "input": "image 640 px -> detections -> max score per class"}

    from_dets("yolo_final", a.yolo_val, a.yolo_test, a.yolo_label)
    from_dets("svm_baseline", a.svm_val, a.svm_test, a.svm_label)

    if a.vit_weights and not a.skip_vit:
        import vit_train
        sv = pd.read_csv(Path(cfg["outputs_dir"]) / "q6" / "vit" / a.vit_name / "val_scores.csv", index_col="image_id").reindex(ids_v)
        sv.columns = cls_ids
        st, lat = vit_train.score_split(cfg, a.vit_weights, "test", allow_test=True)
        lock_info["weights_md5"]["vit"] = md5(a.vit_weights)
        models["vit"] = {"sv": sv, "st": st.reindex(ids_t), "lat": lat["median_ms"], "hw": lat["hardware"],
                         "train_min": logged_value(cfg, a.vit_name, "train_time_min"), "params": 86_000_000, "input": f"letterbox {lat['input']}x{lat['input']}, 16x16 patches"}
    if a.clip:
        import clip_zeroshot as C
        chosen = json.load(open(out / "clip_chosen.json"))
        prompts = C.load_prompts()
        scorer = C.get_scorer(a.clip_model)
        sv = C.clip_scores(cfg, scorer, prompts, chosen, "val").reindex(ids_v)
        st = C.clip_scores(cfg, scorer, prompts, chosen, "test", allow_test=True).reindex(ids_t)
        lat = json.load(open(out / "clip_latency.json")) if (out / "clip_latency.json").exists() else C.measure_latency(cfg, scorer, prompts, chosen)
        models["clip"] = {"sv": sv, "st": st, "lat": lat["median_ms"], "hw": lat["hardware"], "train_min": None, "params": 151_000_000,
                          "input": f"{chosen['variant']} tiles, prompts {chosen['prompt_set']}, {chosen['aggregator']}"}
    if not models:
        raise SystemExit("no model given")

    rows, per_class, preds_t = [], {}, {}
    for k, m in models.items():
        thr = L.tune_thresholds(m["sv"], y_v)                                    # thresholds from VALIDATION
        met = L.metrics(m["st"], y_t, thr)
        pred = L.predict_matrix(m["st"][cls_ids], thr)
        preds_t[k] = pred
        lo, hi = L.bootstrap_macro_f1(pred, y_t[cls_ids].values.astype(bool), a.bootstrap)
        t_any = L.any_operating_point(m["sv"], y_v, a.target_recall)
        op = L.any_at(m["st"], y_t, t_any)
        mv = L.metrics(m["sv"], y_v, thr)
        per_class[k] = met["per_class"].set_index("cls")["f1"]
        q = meta.get({"yolo_final": "yolo_final", "svm_baseline": "svm_baseline", "vit": "vit", "clip": "clip"}[k], {})
        cost = ""
        if a.gpu_usd_per_hour and m["lat"]:
            cost = f"{float(m['lat']) / 3.6e6 * a.gpu_usd_per_hour * 1000:.5f} USD per 1000 images (assumed {a.gpu_usd_per_hour} USD/h, latency x rate)"
        rows.append({"model": q.get("label", k), "key": k, "task_input": m["input"], "test_macro_F1": round(met["macro_f1"], 4), "macro_F1_CI95": f"[{lo:.3f}, {hi:.3f}]",
                     "test_macro_AP": round(met["macro_ap"], 4), "val_macro_F1": round(mv["macro_f1"], 4),
                     f"defect_recall@val_op({a.target_recall:.0%})": round(op["recall"], 3), "false_alarm_rate@val_op": round(op["false_alarm_rate"], 3),
                     "latency_ms_median": None if m["lat"] is None else round(float(m["lat"]), 1), "hardware": m["hw"], "train_time_min": m["train_min"],
                     "labelled_training_data": q.get("labelled_training_images", ""), "third_party_dependence": q.get("third_party", ""),
                     "privacy": q.get("privacy", ""), "marginal_cost": cost or q.get("marginal_cost_note", "")})
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "q6_comparison.csv", index=False)
    rec_col = f"defect_recall@val_op({a.target_recall:.0%})"
    (out / "q6_comparison.md").write_text(table_md(comp[["model", "test_macro_F1", "macro_F1_CI95", "test_macro_AP", rec_col, "false_alarm_rate@val_op",
                                                         "latency_ms_median", "hardware", "train_time_min", "labelled_training_data"]]))
    pc = pd.DataFrame(per_class)
    pc.index = [names[c] for c in pc.index]
    pc.round(3).to_csv(out / "q6_per_class_f1.csv")

    ref = a.reference or next((k for k in ("yolo_final", "svm_baseline") if k in models), list(models)[0])
    yt = y_t[cls_ids].values.astype(bool)
    pairs = []
    for k in models:
        if k != ref:
            d = L.paired_bootstrap_diff(preds_t[k], preds_t[ref], yt, 2000)
            pairs.append({"model_A": k, "model_B(reference)": ref, "macro_F1_A_minus_B": round(d["mean_diff"], 4), "CI95": [round(x, 4) for x in d["ci95"]], "P(A better)": round(d["p_a_better"], 3)})
    pd.DataFrame(pairs).to_csv(out / "q6_paired_bootstrap.csv", index=False)
    json.dump({"comparison": rows, "paired": pairs, "reference": ref, "n_test_images": len(ids_t), "n_defective": int(y_t.any(axis=1).sum()),
               "hardware_now": hardware_string()}, open(out / "q6_final.json", "w"), indent=2, default=str)
    lock_info["args"] = vars(a)
    json.dump(lock_info, open(lock, "w"), indent=2, default=str)
    print(comp[["model", "test_macro_F1", "macro_F1_CI95", "test_macro_AP", "latency_ms_median"]].to_string(index=False))
    print("\nPaired bootstrap (macro-F1 difference vs reference):\n" + pd.DataFrame(pairs).to_string(index=False))
    print(f"\n-> {out}  (lock: {lock})\nA difference whose 95% CI contains 0 is NOT a win: say so in the report.")


if __name__ == "__main__":
    main()
