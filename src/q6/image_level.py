"""
Image-level scoring shared by ViT, CLIP, YOLO and the SVM baseline (Q6).

Why image level: ViT and CLIP have no boxes, so mAP@0.5 (the detection metric) cannot be computed for them. The comparable
metric is therefore image-level, multi-label: "is defect class c present in this image?", scored by MACRO-F1 over the 7 classes with
per-class thresholds tuned on VALIDATION only. Detectors are converted by taking, per class, the maximum detection confidence in the
image. Also reported: macro average precision (threshold-free), and the defective-vs-defect-free decision at a validation-chosen point.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def multi_hot(gt, image_ids, class_ids):
    y = pd.DataFrame(False, index=list(image_ids), columns=list(class_ids))
    if len(gt):
        for iid, c in gt[["image_id", "cls"]].drop_duplicates().itertuples(index=False):
            if iid in y.index and c in y.columns:
                y.at[iid, c] = True
    return y


def scores_from_detections(preds, image_ids, class_ids):
    s = pd.DataFrame(0.0, index=list(image_ids), columns=list(class_ids))
    if len(preds):
        for (iid, c), v in preds.groupby(["image_id", "cls"])["conf"].max().items():
            if iid in s.index and c in s.columns:
                s.at[iid, c] = float(v)
    return s


def tune_thresholds(scores, y, min_thr=1e-6):
    """Per class: threshold maximising F1 on the given (VALIDATION) data. A score of 0 never counts as a detection."""
    thr = {}
    for c in scores.columns:
        yt, sc = y[c].values.astype(int), scores[c].values
        if yt.sum() == 0:
            thr[c] = float("inf")
            continue
        p, r, t = precision_recall_curve(yt, sc)
        f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
        thr[c] = max(float(t[int(np.argmax(f1))]), min_thr)
    return thr


def predict_matrix(scores, thr):
    return np.stack([scores[c].values >= thr[c] for c in scores.columns], 1)


def _f1(tp, fp, fn):
    p = tp / np.maximum(tp + fp, 1)
    r = tp / np.maximum(tp + fn, 1)
    return 2 * p * r / np.maximum(p + r, 1e-12), p, r


def macro_f1_from(pred, yt):
    tp, fp, fn = (pred & yt).sum(0), (pred & ~yt).sum(0), (~pred & yt).sum(0)
    f1, _, _ = _f1(tp, fp, fn)
    has = yt.sum(0) > 0
    return float(f1[has].mean())


def metrics(scores, y, thr):
    """Returns dict with macro_f1 (primary), per-class table, macro_ap, macro_auc, any-defect recall/FPR at the class thresholds."""
    yt, pred = y[scores.columns].values.astype(bool), predict_matrix(scores, thr)
    tp, fp, fn = (pred & yt).sum(0), (pred & ~yt).sum(0), (~pred & yt).sum(0)
    f1, p, r = _f1(tp, fp, fn)
    has = yt.sum(0) > 0
    ap = [average_precision_score(yt[:, i], scores.iloc[:, i].values) if has[i] else np.nan for i in range(yt.shape[1])]
    auc = [roc_auc_score(yt[:, i], scores.iloc[:, i].values) if has[i] and (~yt[:, i]).any() else np.nan for i in range(yt.shape[1])]
    per = pd.DataFrame({"cls": list(scores.columns), "support": yt.sum(0), "precision": p, "recall": r, "f1": f1, "ap": ap, "auc": auc})
    any_t, any_p = yt.any(1), pred.any(1)
    return {"macro_f1": float(f1[has].mean()), "macro_ap": float(np.nanmean(ap)), "macro_auc": float(np.nanmean(auc)), "per_class": per,
            "any_defect_recall_at_class_thr": float((any_t & any_p).sum() / max(any_t.sum(), 1)),
            "any_defect_fpr_at_class_thr": float((~any_t & any_p).sum() / max((~any_t).sum(), 1)),
            "n_images": int(len(y)), "n_defective": int(any_t.sum())}


def any_operating_point(scores_val, y_val, target_recall=0.95):
    """Highest threshold on max-over-classes score that reaches the target recall of defective images (VALIDATION)."""
    s, t = scores_val.max(axis=1).values, y_val.any(axis=1).values
    cands = np.unique(s[t]) if t.any() else np.array([0.0])
    ok = [c for c in cands if (s[t] >= c).mean() >= target_recall]
    return float(max(ok)) if ok else float(cands.min())


def any_at(scores, y, thr):
    s, t = scores.max(axis=1).values, y.any(axis=1).values
    flag = s >= thr
    return {"recall": float((flag & t).sum() / max(t.sum(), 1)), "false_alarm_rate": float((flag & ~t).sum() / max((~t).sum(), 1)), "thr": float(thr)}


def bootstrap_macro_f1(pred, yt, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(yt)
    idx = rng.integers(0, N, size=(n, N))
    tp, fp, fn = (pred & yt)[idx].sum(1), (pred & ~yt)[idx].sum(1), (~pred & yt)[idx].sum(1)
    f1, _, _ = _f1(tp, fp, fn)
    has = yt[idx].sum(1) > 0
    m = np.where(has, f1, np.nan)
    with warnings.catch_warnings():                       # a resample with no positives gives NaN: ignore those resamples
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.nanmean(m, axis=1)
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def paired_bootstrap_diff(pred_a, pred_b, yt, n=2000, seed=0):
    """macro-F1(A) - macro-F1(B) on resampled images (same resample for both). Returns mean, 95% CI, P(A > B)."""
    rng = np.random.default_rng(seed)
    N = len(yt)
    idx = rng.integers(0, N, size=(n, N))
    out = []
    for p in (pred_a, pred_b):
        tp, fp, fn = (p & yt)[idx].sum(1), (p & ~yt)[idx].sum(1), (~p & yt)[idx].sum(1)
        f1, _, _ = _f1(tp, fp, fn)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out.append(np.nanmean(np.where(yt[idx].sum(1) > 0, f1, np.nan), axis=1))
    d = out[0] - out[1]
    return {"mean_diff": float(np.nanmean(d)), "ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))], "p_a_better": float(np.nanmean(d > 0))}
