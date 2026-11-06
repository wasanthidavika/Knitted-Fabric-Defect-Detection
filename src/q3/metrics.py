"""
Shared detection metrics. ONE implementation scores YOLO, the SVM baseline and the trivial benchmark,
so the comparison in Q3.a cannot be skewed by different mAP code.

Prediction format (DataFrame): image_id, cls, conf, x1, y1, x2, y2   (coordinates normalised 0..1)
Ground-truth format:           image_id, cls, x1, y1, x2, y2

AP = COCO-style 101-point interpolated AP. Ultralytics' own mAP uses a slightly different integration, so numbers can
differ by a small amount; report the difference if you quote both.
"""
import numpy as np
import pandas as pd

IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)
PRED_COLS = ["image_id", "cls", "conf", "x1", "y1", "x2", "y2"]
GT_COLS = ["image_id", "cls", "x1", "y1", "x2", "y2"]


def iou_matrix(a, b):
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-12)


def nms(boxes, scores, iou_thr=0.5, top_k=None, pre_k=None):
    """Greedy NMS; returns indices of kept boxes (highest score first)."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores), kind="stable")
    if pre_k:
        order = order[:pre_k]
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if top_k and len(keep) >= top_k:
            break
        rest = order[1:]
        if not len(rest):
            break
        order = rest[iou_matrix(boxes[i:i + 1], boxes[rest])[0] <= iou_thr]
    return np.array(keep, dtype=int)


def _ap(conf, tp, n_gt):
    if n_gt == 0:
        return np.nan
    if len(conf) == 0:
        return 0.0
    o = np.argsort(-conf, kind="stable")
    tp = tp[o].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
    q = np.zeros(101)
    ok = idx < len(prec)
    q[ok] = prec[idx[ok]]
    return float(q.mean())


def evaluate_detections(preds, gts, class_ids, iou_thrs=IOU_THRS):
    """Returns dict: map50, map50_95, ap50_per_class, ap_by_iou (class x thr array), n_gt_per_class."""
    preds = preds if len(preds) else pd.DataFrame(columns=PRED_COLS)
    ap = np.full((len(class_ids), len(iou_thrs)), np.nan)
    n_gt = {}
    pbc = {c: g for c, g in preds.groupby("cls")}
    gbc = {c: g for c, g in gts.groupby("cls")}
    for ci, c in enumerate(class_ids):
        g = gbc.get(c)
        n_gt[c] = 0 if g is None else len(g)
        if n_gt[c] == 0:
            continue
        p = pbc.get(c)
        if p is None or not len(p):
            ap[ci, :] = 0.0
            continue
        gt_img = {k: v[["x1", "y1", "x2", "y2"]].values for k, v in g.groupby("image_id")}
        confs, tps = [], []
        for img, pi in p.groupby("image_id"):
            pi = pi.sort_values("conf", ascending=False, kind="stable")
            pb = pi[["x1", "y1", "x2", "y2"]].values
            confs.append(pi["conf"].values)
            tp = np.zeros((len(pb), len(iou_thrs)), bool)
            gb = gt_img.get(img)
            if gb is not None:
                iou = iou_matrix(pb, gb)
                for ti, t in enumerate(iou_thrs):
                    taken = np.zeros(len(gb), bool)
                    for i in range(len(pb)):
                        cand = np.where(taken, -1.0, iou[i])
                        j = int(cand.argmax())
                        if cand[j] >= t:
                            taken[j] = True
                            tp[i, ti] = True
            tps.append(tp)
        conf, tp = np.concatenate(confs), np.concatenate(tps)
        for ti in range(len(iou_thrs)):
            ap[ci, ti] = _ap(conf, tp[:, ti], n_gt[c])
    has = ~np.isnan(ap[:, 0])
    return {"map50": float(np.nanmean(ap[has, 0])) if has.any() else float("nan"),
            "map50_95": float(np.nanmean(ap[has, :].mean(axis=1))) if has.any() else float("nan"),
            "ap50_per_class": {c: (None if np.isnan(ap[i, 0]) else float(ap[i, 0])) for i, c in enumerate(class_ids)},
            "ap_by_iou": ap, "n_gt_per_class": n_gt}


def image_level_scores(preds, gts, image_ids, class_ids, conf_thr):
    """Image-level presence metrics at a confidence threshold.
    Returns (per-class DataFrame, summary dict with macro_f1, any_defect_recall, any_defect_fpr)."""
    image_ids = list(image_ids)
    sel = preds[preds["conf"] >= conf_thr]
    yp = (sel.groupby(["image_id", "cls"]).size().unstack(fill_value=0) > 0)
    yp = yp.reindex(index=image_ids, columns=class_ids, fill_value=False).fillna(False).astype(bool)
    yt = (gts.groupby(["image_id", "cls"]).size().unstack(fill_value=0) > 0)
    yt = yt.reindex(index=image_ids, columns=class_ids, fill_value=False).fillna(False).astype(bool)
    rows = []
    for c in class_ids:
        tp = int((yt[c] & yp[c]).sum()); fp = int((~yt[c] & yp[c]).sum()); fn = int((yt[c] & ~yp[c]).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"cls": c, "support_images": int(yt[c].sum()), "precision": pr, "recall": rc,
                     "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0})
    df = pd.DataFrame(rows)
    any_t, any_p = yt.any(axis=1), yp.any(axis=1)
    summary = {"macro_f1": float(df.loc[df.support_images > 0, "f1"].mean()),
               "any_defect_recall": float((any_t & any_p).sum() / max(any_t.sum(), 1)),
               "any_defect_fpr": float((~any_t & any_p).sum() / (~any_t).sum()) if (~any_t).any() else float("nan"),
               "conf_thr": float(conf_thr)}
    return df, summary


def trivial_preds(image_ids, cls):
    """Reference benchmark: one whole-image box of a fixed class on every image."""
    return pd.DataFrame({"image_id": list(image_ids), "cls": int(cls), "conf": 1.0,
                         "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0})
