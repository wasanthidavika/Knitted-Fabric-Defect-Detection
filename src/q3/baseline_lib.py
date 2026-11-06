"""
Classical baseline: texture-feature sliding-window DETECTOR (Gabor + GLCM + LBP -> RBF-SVM), two-stage cascade.

Pipeline per image (all on the SAME resized images the YOLO models see):
  1. windows: shapes = k-means clusters of the TRAINING boxes (so every defect size has a matching window);
     stride = half the window's shorter side (min 8 px)
  2. stage 1 (cheap, integral images): 34 features = Gabor response mean/std (12 filters x 2) + LBP histogram (10)
     -> linear objectness score -> NMS -> keep top-K windows           (K = 300 by default)
  3. stage 2: + 32 GLCM features (4 properties x 2 distances x 4 angles) = 66 features -> StandardScaler -> RBF-SVM
     over 7 defect classes + background; confidence = sigmoid(best defect score - background score)
  4. ridge box regression (classic R-CNN style) refines each window; per-class NMS; top 100 per image

Why a cascade: GLCM per window is the slow part. Stage 1 keeps the pipeline affordable. It also means the SVM is
trained on the SAME kind of windows it sees at test time (windows chosen by stage 1), not on easy random negatives.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor_kernel
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import PRED_COLS, iou_matrix, nms  # noqa: E402

import pandas as pd  # noqa: E402

FREQS = (0.10, 0.20, 0.35)
THETAS = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS = ("contrast", "homogeneity", "energy", "correlation")
N_GABOR, N_LBP = len(FREQS) * len(THETAS), 10
FEATURE_NAMES = ([f"gabor_mean_{i}" for i in range(N_GABOR)] + [f"gabor_std_{i}" for i in range(N_GABOR)] +
                 [f"lbp_{i}" for i in range(N_LBP)] + [f"glcm_{p}_{d}_{a}" for p in GLCM_PROPS for d in (1, 2) for a in range(4)])
_K = None


def _kernels():
    global _K
    if _K is None:
        _K = []
        for f in FREQS:
            for th in THETAS:
                k = gabor_kernel(f, theta=th)
                _K.append((np.ascontiguousarray(k.real, dtype=np.float32), np.ascontiguousarray(k.imag, dtype=np.float32)))
    return _K


class ImageFeatures:
    """Per-image maps and integral images; window features are then O(1) lookups (cheap) or a small GLCM (stage 2)."""

    def __init__(self, gray):
        self.H, self.W = gray.shape
        self.I, self.I2 = [], []
        for kr, ki in _kernels():
            re = cv2.filter2D(gray, cv2.CV_32F, kr, borderType=cv2.BORDER_REFLECT)
            im = cv2.filter2D(gray, cv2.CV_32F, ki, borderType=cv2.BORDER_REFLECT)
            s, sq = cv2.integral2(np.sqrt(re * re + im * im))
            self.I.append(s)
            self.I2.append(sq)
        lbp = local_binary_pattern((gray * 255).astype(np.uint8), 8, 1, "uniform").astype(np.uint8)
        self.L = [cv2.integral((lbp == b).astype(np.uint8)) for b in range(N_LBP)]
        self.gq = np.minimum((gray * 16).astype(np.uint8), 15)

    @staticmethod
    def _sum(I, b):
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return I[y2, x2].astype(np.float64) - I[y1, x2] - I[y2, x1] + I[y1, x1]

    def cheap(self, boxes):
        """(N, 34): Gabor mean (12), Gabor std (12), LBP histogram (10)."""
        area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).astype(np.float64)
        means, stds = [], []
        for I, I2 in zip(self.I, self.I2):
            m = self._sum(I, boxes) / area
            stds.append(np.sqrt(np.maximum(self._sum(I2, boxes) / area - m * m, 0)))
            means.append(m)
        lbp = [self._sum(L, boxes) / area for L in self.L]
        return np.stack(means + stds + lbp, 1).astype(np.float32)

    def glcm(self, boxes):
        """(N, 32) GLCM texture features on a 16-level quantised crop."""
        out = np.zeros((len(boxes), 32), np.float32)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            g = graycomatrix(self.gq[y1:y2, x1:x2], distances=[1, 2], angles=GLCM_ANGLES, levels=16, symmetric=True, normed=True)
            out[i] = np.concatenate([graycoprops(g, p).ravel() for p in GLCM_PROPS])
        return np.nan_to_num(out)


def load_gray(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(path)
    return g.astype(np.float32) / 255.0


def fit_anchors(wh_px, k=5, seed=42):
    """Window shapes: k-means on log(width, height) of TRAINING boxes (pixels in the resized image)."""
    km = KMeans(k, n_init=10, random_state=seed).fit(np.log(np.maximum(np.asarray(wh_px, float), 4)))
    a = np.exp(km.cluster_centers_)
    return np.round(a[np.argsort(a[:, 0] * a[:, 1])]).astype(int)


def make_windows(H, W, anchors, min_size=8, min_stride=8):
    out = []
    for aw, ah in anchors:
        aw, ah = int(min(max(aw, min_size), W)), int(min(max(ah, min_size), H))
        s = max(min_stride, int(0.5 * min(aw, ah)))
        xs, ys = np.arange(0, W - aw + 1, s), np.arange(0, H - ah + 1, s)
        if not len(xs) or xs[-1] != W - aw:
            xs = np.append(xs, W - aw)
        if not len(ys) or ys[-1] != H - ah:
            ys = np.append(ys, H - ah)
        gx, gy = np.meshgrid(xs, ys)
        out.append(np.stack([gx.ravel(), gy.ravel(), gx.ravel() + aw, gy.ravel() + ah], 1))
    return np.unique(np.concatenate(out), axis=0).astype(np.int32)


def stage1_samples(gray, anchors, gt_px, rng, n_pos=30, n_neg=150):
    """Cheap features + labels (1 = IoU>=0.5 with a defect, 0 = IoU<0.3) of sampled windows of one image."""
    F = ImageFeatures(gray)
    wins = make_windows(F.H, F.W, anchors)
    iou = iou_matrix(wins.astype(float), gt_px).max(1) if len(gt_px) else np.zeros(len(wins))
    pos, neg = np.where(iou >= 0.5)[0], np.where(iou < 0.3)[0]
    pos = rng.choice(pos, min(n_pos, len(pos)), replace=False) if len(pos) else pos
    neg = rng.choice(neg, min(n_neg, len(neg)), replace=False) if len(neg) else neg
    idx = np.r_[pos, neg].astype(int)
    return F.cheap(wins[idx]), np.r_[np.ones(len(pos)), np.zeros(len(neg))]


def extract_candidates(gray, anchors, stage1, K=300, nms_iou=0.5, gt_px=None, gt_cls=None):
    """Stage 1 + stage-2 features for the top-K windows. With ground truth (train only) also IoU/class/targets."""
    F = ImageFeatures(gray)
    wins = make_windows(F.H, F.W, anchors)
    cheap = F.cheap(wins)
    s1 = stage1.decision_function(cheap)
    keep = nms(wins.astype(float), s1, nms_iou, top_k=K, pre_k=4000)
    b = wins[keep]
    out = {"boxes": b, "s1": s1[keep].astype(np.float32),
           "X": np.concatenate([cheap[keep], F.glcm(b)], 1).astype(np.float32), "hw": np.array([F.H, F.W])}
    if gt_px is not None:
        if len(gt_px):
            iou = iou_matrix(b.astype(float), gt_px)
            j = iou.argmax(1)
            g = gt_px[j]
            wcx, wcy, ww, wh = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
            gcx, gcy, gw, gh = (g[:, 0] + g[:, 2]) / 2, (g[:, 1] + g[:, 3]) / 2, np.maximum(g[:, 2] - g[:, 0], 1), np.maximum(g[:, 3] - g[:, 1], 1)
            out.update({"iou_best": iou.max(1).astype(np.float32), "cls_best": np.asarray(gt_cls)[j].astype(np.int16),
                        "targets": np.stack([(gcx - wcx) / ww, (gcy - wcy) / wh, np.log(gw / ww), np.log(gh / wh)], 1).astype(np.float32)})
        else:
            n = len(b)
            out.update({"iou_best": np.zeros(n, np.float32), "cls_best": np.zeros(n, np.int16), "targets": np.zeros((n, 4), np.float32)})
    return out


def windows_to_detections(bundle, image_id, boxes, X, hw, per_class_nms=None, max_per_image=100, regress=True):
    """Stage-2 scoring -> box refinement -> per-class NMS -> detections DataFrame (normalised boxes)."""
    scaler, svm, ridge = bundle["scaler"], bundle["svm"], bundle["ridge"]
    per_class_nms = bundle.get("per_class_nms", 0.5) if per_class_nms is None else per_class_nms
    Xs = scaler.transform(X)
    d = svm.decision_function(Xs)
    classes = svm.classes_
    if d.ndim == 1:
        d = np.stack([-d, d], 1)
    bg = bundle["bg_label"]
    fg = np.where(classes != bg)[0]
    if bg not in classes or not len(fg):
        return pd.DataFrame(columns=PRED_COLS)
    bgcol = int(np.where(classes == bg)[0][0])
    j = d[:, fg].argmax(1)
    cls = classes[fg][j]
    conf = 1.0 / (1.0 + np.exp(-(d[:, fg].max(1) - d[:, bgcol])))
    b = boxes.astype(float)
    if regress and ridge is not None:
        t = ridge.predict(Xs)
        wcx, wcy, ww, wh = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        cx, cy = wcx + np.clip(t[:, 0], -1, 1) * ww, wcy + np.clip(t[:, 1], -1, 1) * wh
        w, h = ww * np.exp(np.clip(t[:, 2], -1, 1)), wh * np.exp(np.clip(t[:, 3], -1, 1))
        b = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    H, W = hw
    bn = np.clip(b / np.array([W, H, W, H]), 0, 1)
    rows = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        keep = idx[nms(bn[idx], conf[idx], per_class_nms)]
        rows += [(image_id, int(c), float(conf[i]), *map(float, bn[i])) for i in keep]
    df = pd.DataFrame(rows, columns=PRED_COLS)
    return df.sort_values("conf", ascending=False).head(max_per_image).reset_index(drop=True)
