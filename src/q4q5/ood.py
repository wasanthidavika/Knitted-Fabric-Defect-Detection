"""
Simple, honest "does this look like the training images?" check for the web app (Q4 out-of-distribution handling).

8 global image statistics -> Mahalanobis distance to the TRAIN distribution; the limit is a high percentile of the training
distances. This catches photos that are clearly not knitted fabric (people, screens, other textures, blank/blurred/very
dark images). It does NOT prove a fabric image is fine and it cannot see subtle shifts (new camera, new yarn).
"""
import cv2
import numpy as np

FEATURE_NAMES = ["mean_r", "mean_g", "mean_b", "gray_std", "log_laplacian_var", "mean_saturation", "log_mean_gradient", "clipped_fraction"]


def features(img_rgb, work=320):
    h, w = img_rgb.shape[:2]
    s = work / max(h, w)
    im = cv2.resize(img_rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img_rgb
    f = im.astype(np.float32) / 255.0
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return np.array([f[..., 0].mean(), f[..., 1].mean(), f[..., 2].mean(), g.std(),
                     np.log1p(cv2.Laplacian(g, cv2.CV_32F).var() * 1e4), hsv[..., 1].mean() / 255.0,
                     np.log1p(np.sqrt(gx ** 2 + gy ** 2).mean() * 1e3), float(((g < 0.02) | (g > 0.98)).mean())])


def fit(F, percentile=99.5, shrinkage=0.1, std_floor=0.02):
    """Standardise features, shrink the covariance towards the identity (robust to the near-collinear R/G/B means of grey
    fabric), distance = Mahalanobis in standardised space, limit = a high percentile of the TRAIN distances."""
    mu = F.mean(0)
    sd = np.maximum(F.std(0), std_floor)
    Z = (F - mu) / sd
    cov = (1 - shrinkage) * np.cov(Z, rowvar=False) + shrinkage * np.eye(F.shape[1])
    stats = {"mean": mu.tolist(), "scale": sd.tolist(), "cov_inv": np.linalg.inv(cov).tolist(), "percentile": percentile,
             "shrinkage": shrinkage, "n_fit": int(len(F)), "features": FEATURE_NAMES}
    d = distances(F, stats)
    stats.update({"limit": float(np.percentile(d, percentile)), "train_distance_median": float(np.median(d))})
    return stats


def distances(F, stats):
    Z = (np.atleast_2d(F) - np.asarray(stats["mean"])) / np.asarray(stats["scale"])
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Z, np.asarray(stats["cov_inv"]), Z), 0))


def score(img_rgb, stats):
    d = float(distances(features(img_rgb), stats)[0])
    return {"distance": d, "limit": stats["limit"], "out_of_distribution": bool(d > stats["limit"])}
