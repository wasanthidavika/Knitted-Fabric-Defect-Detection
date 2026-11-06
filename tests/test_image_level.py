"""Unit tests for image_level.py and vit_facts.py (no torch needed). Run: python tests/test_image_level.py"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import image_level as L
import vit_facts as V

ids = [f"i{k}" for k in range(8)]
gt = pd.DataFrame({"image_id": ["i0", "i0", "i1", "i2", "i3"], "cls": [0, 1, 0, 1, 0]})
y = L.multi_hot(gt, ids, [0, 1])
assert y.loc["i0"].all() and y.loc["i1", 0] and not y.loc["i1", 1] and not y.loc["i7"].any()

# perfect detector -> macro F1 1; scores from detections take the per-class max
preds = pd.DataFrame({"image_id": ["i0", "i0", "i0", "i1", "i2", "i3"], "cls": [0, 0, 1, 0, 1, 0], "conf": [.9, .5, .8, .7, .6, .95]})
s = L.scores_from_detections(preds, ids, [0, 1])
assert s.loc["i0", 0] == 0.9 and s.loc["i4", 0] == 0.0
thr = L.tune_thresholds(s, y)
m = L.metrics(s, y, thr)
assert abs(m["macro_f1"] - 1) < 1e-9 and abs(m["macro_ap"] - 1) < 1e-9, m["macro_f1"]
assert all(t > 0 for t in thr.values())                                       # zero scores never count as positives

# thresholds tuned on val are FIXED for test: a worse test scoring must lower F1
s_bad = s.copy(); s_bad.loc["i3", 0] = 0.0
assert L.metrics(s_bad, y, thr)["macro_f1"] < 1

# class with no validation positives -> never predicted, excluded from macro average
y2 = y.copy(); y2[1] = False
assert np.isinf(L.tune_thresholds(s, y2)[1])

# any-defect operating point
t_any = L.any_operating_point(s, y, 1.0); a = L.any_at(s, y, t_any)
assert a["recall"] == 1.0 and a["false_alarm_rate"] == 0.0

# bootstrap: CI contains the point estimate; identical models -> diff 0; clearly better model -> P(A>B) ~ 1
pred = L.predict_matrix(s, thr); yt = y.values
lo, hi = L.bootstrap_macro_f1(pred, yt, n=500); assert lo <= 1.0 <= hi + 1e-9
d0 = L.paired_bootstrap_diff(pred, pred, yt, n=300); assert abs(d0["mean_diff"]) < 1e-12
worse = ~pred
d1 = L.paired_bootstrap_diff(pred, worse, yt, n=300); assert d1["p_a_better"] > 0.95 and d1["mean_diff"] > 0.3

# ViT arithmetic: tokens/attention/params match the well-known numbers
assert V.n_tokens(384, 16) == 576 and V.n_tokens(224, 16) == 196 and V.n_tokens(224, 32) == 49
assert V.attention_entries(577) == 332_929
assert V.vit_params(224, classes=1000) == 86_567_656                         # published ViT-B/16 parameter count
assert V.vit_params(384, classes=7) < 87_500_000
lb = V.letterbox_square(np.zeros((300, 600, 3), np.uint8) + 7, 384)
assert lb.shape == (384, 384, 3) and lb[0, 0, 0] == 114 and lb[192, 192, 0] == 7
print("image_level / vit_facts unit tests: all passed")
