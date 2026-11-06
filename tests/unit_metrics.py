"""Unit tests for src/metrics.py (run: python tests/unit_metrics.py)."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "q3"))
from metrics import evaluate_detections, iou_matrix, nms, image_level_scores, trivial_preds

gt = pd.DataFrame({"image_id": ["a", "a", "b"], "cls": [0, 1, 0],
                   "x1": [.1, .6, .2], "y1": [.1, .6, .2], "x2": [.3, .8, .4], "y2": [.3, .8, .4]})
def P(rows): return pd.DataFrame(rows, columns=["image_id", "cls", "conf", "x1", "y1", "x2", "y2"])
ids = [0, 1]

assert abs(iou_matrix([[0, 0, 1, 1]], [[0, 0, .5, 1]])[0, 0] - .5) < 1e-9

perfect = P([[r.image_id, r.cls, .9, r.x1, r.y1, r.x2, r.y2] for r in gt.itertuples()])
r = evaluate_detections(perfect, gt, ids); assert abs(r["map50"] - 1) < 1e-9 and abs(r["map50_95"] - 1) < 1e-9, r["map50"]

wrong = perfect.copy(); wrong["cls"] = 1 - wrong["cls"]
assert evaluate_detections(wrong, gt, ids)["map50"] == 0.0
assert evaluate_detections(P([]), gt, ids)["map50"] == 0.0

shift = perfect.copy(); shift[["x1", "x2"]] += 0.04      # IoU ~0.67: ok at 0.5, fails at strict thresholds
r = evaluate_detections(shift, gt, ids); assert r["map50"] == 1.0 and r["map50_95"] < 1.0

one = gt[gt.cls == 1]
fp_first = P([["a", 1, .9, .0, .0, .05, .05], ["a", 1, .5, .6, .6, .8, .8]])     # FP ranked above the TP
ap = evaluate_detections(fp_first, one, [1])["ap50_per_class"][1]; assert abs(ap - .5) < .02, ap
tp_first = P([["a", 1, .9, .6, .6, .8, .8], ["a", 1, .5, .0, .0, .05, .05]])
assert abs(evaluate_detections(tp_first, one, [1])["ap50_per_class"][1] - 1) < 1e-9

dup = P([["b", 0, .9, .2, .2, .4, .4], ["b", 0, .8, .2, .2, .4, .4]])             # duplicate of same object
assert abs(evaluate_detections(dup, gt[gt.image_id == "b"], [0])["ap50_per_class"][0] - 1) < 1e-9

assert list(nms([[0, 0, 1, 1], [0, 0, .95, .95], [2, 2, 3, 3]], [.9, .8, .7], .5)) == [0, 2]

df, s = image_level_scores(perfect, gt, ["a", "b", "c"], ids, .5)
assert s["any_defect_recall"] == 1.0 and s["any_defect_fpr"] == 0.0 and abs(s["macro_f1"] - 1) < 1e-9
t = trivial_preds(["a", "b"], 0); assert len(t) == 2 and (t.x2 == 1).all()
print("metrics unit tests: all passed")
