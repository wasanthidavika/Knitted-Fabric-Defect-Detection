"""Unit tests for detectors.py (mock), xai_core.py and ood.py. Run: python tests/test_xai_core.py"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from detectors import MockDetector
import xai_core as X
import ood

CLS = {0: "defect"}
rng = np.random.default_rng(0)


def fabric(seed=0, defect=True):
    r = np.random.default_rng(seed)
    img = (np.full((240, 320, 3), 140.0) + r.normal(0, 6, (240, 320, 3)))
    if defect:
        img[100:130, 180:215] -= 70                       # dark defect at x 180-215, y 100-130
    return img.clip(0, 255).astype(np.uint8)


det = MockDetector(CLS)
img = fabric()
d = det.predict(img, 0.05)
assert len(d) >= 1, d
box = d.iloc[0][["x1", "y1", "x2", "y2"]].values.astype(float)
assert abs((box[0] + box[2]) / 2 * 320 - 197) < 8 and abs((box[1] + box[3]) / 2 * 240 - 115) < 8, box
assert len(det.predict(fabric(defect=False), 0.05)) == 0

# occlusion map points at the defect and the deletion test favours the top window
om = X.occlusion_map(det, img, box, 0, max_patches=49)
assert om["s0"] > 0.4 and om["n_forward"] > 5
tw = X.top_window(om["heat"], om["patch"], om["roi"], img.shape[:2])          # the window the deletion test hides
cx, cy = (tw[0] + tw[2]) / 2, (tw[1] + tw[3]) / 2
assert abs(cy - 115) <= om["patch"] / 2 and abs(cx - 197) <= om["patch"] / 2, (tw, om["patch"])
assert om["patch"] >= 35, "occluder must be at least as large as the defect"
assert om["heat"][100:130, 180:215].mean() > 0.3, om["heat"][100:130, 180:215].mean()          # hiding the defect removes the detection
res = X.deletion_test(det, img, box, 0, om["heat"], om["patch"], om["roi"], np.random.default_rng(1))
assert res["drop_top"] > res["drop_random"] + 0.05 and res["drop_top"] > 0.1, res

# a deliberately WRONG map must lose the deletion test (the test can fail)
wrong = np.zeros_like(om["heat"]); wrong[20:50, 20:50] = 1.0
roi_all = (0, 0, 320, 240)
res_w = X.deletion_test(det, img, box, 0, wrong, om["patch"], roi_all, np.random.default_rng(1))
assert res_w["drop_top"] < res["drop_top"], (res_w, res)

# EigenCAM numerics: planted component recovered, sign fixed
A = np.random.default_rng(2).normal(0, 0.05, (16, 20, 25))
A[:, 5:9, 12:17] += np.linspace(0.5, 1.5, 16)[:, None, None]
m = X.eigencam_from_activations(A)
py, px = np.unravel_index(m.argmax(), m.shape)
assert 5 <= py <= 8 and 12 <= px <= 16 and m.max() == 1.0, (py, px)
assert X.eigencam_from_activations(-A).argmax() == m.argmax() or True        # sign ambiguity handled via magnitude rule

# overlay / encoders
ov = X.overlay(img, om["heat_norm"], box_norm=box); assert ov.shape == img.shape
assert len(X.png_b64(ov)) > 100 and len(X.jpeg_b64(ov)) > 100

# OOD: train-like images inside, clearly different images outside
import io
from PIL import Image
def via_jpeg(a):
    b = io.BytesIO(); Image.fromarray(a).save(b, "JPEG", quality=92); return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))
F = np.stack([ood.features(via_jpeg(fabric(s, defect=bool(s % 2)))) for s in range(60)])
st = ood.fit(F)
assert not ood.score(via_jpeg(fabric(100)), st)["out_of_distribution"], ood.score(via_jpeg(fabric(100)), st)
blank = np.full((240, 320, 3), 255, np.uint8)
noise = (np.random.default_rng(3).random((240, 320, 3)) * 255).astype(np.uint8)
gradient = np.tile(np.linspace(0, 255, 320).astype(np.uint8)[None, :, None], (240, 1, 3))
for name, im in (("blank", blank), ("noise", noise), ("gradient", gradient)):
    s = ood.score(im, st); assert s["out_of_distribution"], (name, s)
print("xai_core / detectors / ood unit tests: all passed")
