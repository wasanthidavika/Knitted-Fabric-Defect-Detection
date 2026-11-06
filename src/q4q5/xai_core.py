"""
Explanation methods for the defect detector (Q5). Everything works per DETECTION, not per image.

1. occlusion_map  : model-agnostic. Hide one patch at a time (blurred fill) around the detection and record how much the
                    detector's confidence for THAT detection falls. Works with any backend. Cost = number of patches forward passes.
2. eigencam       : architecture-specific (CNN backbone/neck), gradient-free, class-agnostic. First principal component of a
                    feature map. One forward pass. Coarse: it shows where features vary most, not why this class was chosen.
3. deletion_test  : sanity check. Hide the most highlighted window vs random windows of the same size and compare the drop in
                    confidence. A faithful map should give the larger drop for its own top window.

Limits to state in the report: occluding creates artificial edges (fill choice matters), results depend on patch size, the
maps are sensitivity maps, NOT proof of the model's reasoning or of causality.
"""
import base64
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import iou_matrix


# ----------------------------------------------------------------------------- target score
def target_score(dets, box_norm, cls, iou_min=0.3):
    """Confidence the detector gives to THE target detection: max conf among detections of class `cls`
    (any class if cls < 0) overlapping the target box with IoU >= iou_min; 0 if none."""
    if dets is None or len(dets) == 0:
        return 0.0
    d = dets if cls < 0 else dets[dets["cls"] == cls]
    if len(d) == 0:
        return 0.0
    iou = iou_matrix(np.asarray(box_norm, float)[None], d[["x1", "y1", "x2", "y2"]].values)[0]
    ok = iou >= iou_min
    return float(d["conf"].values[ok].max()) if ok.any() else 0.0


# ----------------------------------------------------------------------------- occlusion
def occlude(img, region, fill="inpaint"):
    """Hide a window. 'inpaint' (default) re-creates it from its surroundings (Telea), so the defect really disappears;
    'blur' keeps a smudge of the defect; 'mean' paints a flat colour (can itself look like a defect)."""
    x1, y1, x2, y2 = [int(v) for v in region]
    out = img.copy()
    if fill == "mean":
        out[y1:y2, x1:x2] = img.reshape(-1, 3).mean(0).astype(np.uint8)
    elif fill == "blur":
        k = max(31, (max(img.shape[:2]) // 8) | 1)
        out[y1:y2, x1:x2] = cv2.GaussianBlur(img, (0, 0), k / 3)[y1:y2, x1:x2]
    else:
        H, W = img.shape[:2]
        m = 8
        a, b, c, d = max(0, y1 - m), max(0, x1 - m), min(H, y2 + m), min(W, x2 + m)
        crop = np.ascontiguousarray(img[a:c, b:d])
        mask = np.zeros(crop.shape[:2], np.uint8)
        mask[y1 - a:y2 - a, x1 - b:x2 - b] = 255
        out[a:c, b:d] = cv2.inpaint(crop, mask, 3, cv2.INPAINT_TELEA)
    return out


def _grid(roi, p, max_patches):
    rx1, ry1, rx2, ry2 = roi
    stride = max(4, p // 2)
    while True:
        xs = list(range(rx1, max(rx1, rx2 - p) + 1, stride))
        ys = list(range(ry1, max(ry1, ry2 - p) + 1, stride))
        if len(xs) * len(ys) <= max_patches or stride > 4 * p:
            return xs, ys, stride
        stride = int(stride * 1.25) + 1


def window_geometry(shape, box_norm, patch_frac=1.25, margin_patches=1.0, min_patch=12):
    """Occluder size p (px) = 1.25 x the longer box side (so one window CAN hide the whole defect) and the region of
    interest (box +/- 1 window) that is explored."""
    H, W = shape
    bx = np.asarray(box_norm, float) * [W, H, W, H]
    p = int(np.clip(round(patch_frac * max(bx[2] - bx[0], bx[3] - bx[1])), min_patch, max(min_patch, min(H, W) // 2)))
    m = int(margin_patches * p)
    roi = (int(max(0, bx[0] - m)), int(max(0, bx[1] - m)), int(min(W, bx[2] + m)), int(min(H, bx[3] + m)))
    return p, roi


def occlusion_map(detector, img, box_norm, cls, patch_frac=1.25, margin_patches=1.0, max_patches=64, fill="inpaint",
                  min_patch=12, base_dets=None):
    """Returns dict: heat (HxW raw drops, 0 outside ROI), heat_norm (0..1), s0, patch, roi, n_forward."""
    H, W = img.shape[:2]
    p, roi = window_geometry((H, W), box_norm, patch_frac, margin_patches, min_patch)
    xs, ys, stride = _grid(roi, p, max_patches)
    dets0 = base_dets if base_dets is not None else detector.predict(img, 0.001)
    s0 = target_score(dets0, box_norm, cls)
    pos = [(x, y) for y in ys for x in xs]
    bxp = np.asarray(box_norm, float) * [W, H, W, H]                       # always include the window centred on the box itself
    cpos = (int(np.clip((bxp[0] + bxp[2]) / 2 - p // 2, 0, max(0, W - p))), int(np.clip((bxp[1] + bxp[3]) / 2 - p // 2, 0, max(0, H - p))))
    if cpos not in pos:
        pos.append(cpos)
    imgs = [occlude(img, (x, y, min(W, x + p), min(H, y + p)), fill) for x, y in pos]
    scores = [target_score(d, box_norm, cls) for d in detector.predict_batch(imgs, 0.001)] if imgs else []
    heat, cnt = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    for (x, y), s in zip(pos, scores):
        heat[y:y + p, x:x + p] += s0 - s
        cnt[y:y + p, x:x + p] += 1
    heat = np.where(cnt > 0, heat / np.maximum(cnt, 1), 0).astype(np.float32)
    hn = np.clip(heat, 0, None)
    hn = hn / hn.max() if hn.max() > 0 else hn
    return {"heat": heat, "heat_norm": hn, "s0": s0, "patch": p, "roi": roi, "n_forward": len(pos) + 1, "stride": stride}


def top_window(heat, p, roi, shape):
    """p x p window centred on the peak of the (lightly smoothed) heat map, restricted to the ROI.
    The map is already averaged over occluding patches, so it is NOT box-summed again (that would pull the peak away)."""
    H, W = shape
    rx1, ry1, rx2, ry2 = roi
    sm = cv2.GaussianBlur(np.clip(heat, 0, None).astype(np.float32), (0, 0), max(1.0, p / 4))
    mask = np.zeros_like(sm)
    mask[ry1:ry2, rx1:rx2] = 1
    sm = sm * mask
    cy, cx = np.unravel_index(int(np.argmax(sm)), sm.shape)
    x1, y1 = int(np.clip(cx - p // 2, 0, max(0, W - p))), int(np.clip(cy - p // 2, 0, max(0, H - p)))
    return (x1, y1, min(W, x1 + p), min(H, y1 + p))


def random_windows(rng, roi, p, n, avoid, shape, max_iou=0.2, tries=200):
    H, W = shape
    rx1, ry1, rx2, ry2 = roi
    out = []
    for _ in range(tries):
        if len(out) >= n:
            break
        x = int(rng.integers(rx1, max(rx1, min(W - p, rx2 - p)) + 1))
        y = int(rng.integers(ry1, max(ry1, min(H - p, ry2 - p)) + 1))
        w = (x, y, min(W, x + p), min(H, y + p))
        if iou_matrix(np.array(w, float)[None], np.array(avoid, float)[None])[0, 0] <= max_iou:
            out.append(w)
    return out


def deletion_test(detector, img, box_norm, cls, heat, p, roi, rng, n_random=5, fill="inpaint", s0=None):
    """Confidence drop after hiding the top window of `heat` vs the mean drop after hiding random windows of the same size."""
    shape = img.shape[:2]
    if s0 is None:
        s0 = target_score(detector.predict(img, 0.001), box_norm, cls)
    top = top_window(heat, p, roi, shape)
    rnd = random_windows(rng, roi, p, n_random, top, shape)
    imgs = [occlude(img, top, fill)] + [occlude(img, w, fill) for w in rnd]
    sc = [target_score(d, box_norm, cls) for d in detector.predict_batch(imgs, 0.001)]
    return {"s0": s0, "score_top_hidden": sc[0], "drop_top": s0 - sc[0],
            "drop_random": float(s0 - np.mean(sc[1:])) if len(sc) > 1 else float("nan"), "n_random": len(sc) - 1,
            "window": top, "n_forward": len(imgs)}


# ----------------------------------------------------------------------------- EigenCAM
def eigencam_from_activations(A):
    """A: (C,h,w) activations -> (h,w) map in [0,1]: projection on the first principal component, sign fixed so that the
    map correlates positively with the activation magnitude, negatives clipped."""
    C, h, w = A.shape
    X = A.reshape(C, -1).T.astype(np.float64)
    X = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    m = X @ Vt[0]
    mag = np.abs(A).mean(0).ravel()
    if np.corrcoef(m, mag)[0, 1] < 0:
        m = -m
    m = np.clip(m, 0, None).reshape(h, w)
    return (m / m.max() if m.max() > 0 else m).astype(np.float32)


def eigencam_heatmap(detector, img, layer_idx=15):
    """EigenCAM at full image size. Accounts for Ultralytics' letterbox padding of the network input."""
    A, (Hin, Win) = detector.activation(img, layer_idx)
    small = eigencam_from_activations(A)
    big = cv2.resize(small, (Win, Hin), interpolation=cv2.INTER_LINEAR)
    H, W = img.shape[:2]
    r = min(detector.imgsz / H, detector.imgsz / W)
    nh, nw = int(round(H * r)), int(round(W * r))
    top, left = max(0, (Hin - nh) // 2), max(0, (Win - nw) // 2)
    crop = big[top:top + nh, left:left + nw]
    return cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)


def explain_detection(detector, img, box_norm, cls, methods=("occlusion", "eigencam"), eigencam_layer=15, rng=None,
                      max_patches=36, n_random=5, fill="inpaint", base_dets=None):
    """Explain ONE detection with the requested methods and run the deletion sanity check for each.
    Returns {"s0", "occlusion": {...}, "eigencam": {...}}; each method dict has heat_norm, drop_top, drop_random, window..."""
    rng = rng or np.random.default_rng(0)
    H, W = img.shape[:2]
    p, roi = window_geometry((H, W), box_norm)
    dets0 = base_dets if base_dets is not None else detector.predict(img, 0.001)
    s0 = target_score(dets0, box_norm, cls)
    out = {"s0": s0, "patch": p, "roi": roi}
    if s0 <= 0:
        out["note"] = "the detector gives this box no score, so occlusion effects cannot be measured"
    if "occlusion" in methods:
        if s0 > 0:
            om = occlusion_map(detector, img, box_norm, cls, max_patches=max_patches, fill=fill, base_dets=dets0)
            dt = deletion_test(detector, img, box_norm, cls, om["heat"], om["patch"], om["roi"], rng, n_random, fill, s0=s0)
            out["occlusion"] = {"available": True, "heat": om["heat"], "heat_norm": om["heat_norm"], "patch": om["patch"],
                                "n_forward": om["n_forward"] + dt["n_forward"], **dt}
        else:
            out["occlusion"] = {"available": False, "reason": out["note"]}
    if "eigencam" in methods:
        if not getattr(detector, "supports_eigencam", False):
            out["eigencam"] = {"available": False, "reason": f"EigenCAM needs the YOLO (CNN) backend; this backend is '{detector.name}'"}
        else:
            try:
                heat = eigencam_heatmap(detector, img, eigencam_layer)
                e = {"available": True, "heat": heat, "heat_norm": heat, "layer": eigencam_layer}
                if s0 > 0:
                    e.update(deletion_test(detector, img, box_norm, cls, heat, p, roi, rng, n_random, fill, s0=s0))
                out["eigencam"] = e
            except Exception as exc:
                out["eigencam"] = {
                    "available": False,
                    "reason": f"EigenCAM unavailable: {type(exc).__name__}: {exc}"
                }
    return out


# ----------------------------------------------------------------------------- display helpers
def overlay(img_rgb, heat_norm, alpha=0.45, box_norm=None):
    cm = cv2.applyColorMap((np.clip(heat_norm, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    out = (img_rgb * (1 - alpha * (heat_norm[..., None] > 0.05)) + cm * alpha * (heat_norm[..., None] > 0.05)).astype(np.uint8)
    if box_norm is not None:
        H, W = img_rgb.shape[:2]
        x1, y1, x2, y2 = (np.asarray(box_norm) * [W, H, W, H]).astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return out


def png_b64(img_rgb):
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(img_rgb[..., ::-1]))
    return base64.b64encode(buf.tobytes()).decode()


def jpeg_b64(img_rgb, quality=88):
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img_rgb[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()
