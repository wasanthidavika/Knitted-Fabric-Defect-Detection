"""Loads the model and everything the app needs once at start-up; turns detector output into a decision for the inspector."""
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ood as ood_mod                                   # noqa: E402
from common import load_config                          # noqa: E402
from detectors import build_detector                    # noqa: E402

LIMITATIONS = [
    "The model was trained only on knitted-fabric photos from the ISL-Knit dataset and knows 7 defect types. Other fabrics, cameras, lighting or yarns may behave differently.",
    "Small defects (a few pixels across at the model's working size) can be missed. 'No defect found' does NOT mean the fabric is defect-free.",
    "The score is the detector's confidence score, not a probability. The 'seen in validation' figure shows how often detections with similar scores were correct on held-out validation images; on new fabric it can be lower.",
    "Grey and dyed fabrics were not guaranteed to perform equally well (colour was estimated, not recorded in the dataset).",
    "Highlighted areas in the explanation show where the model is sensitive. They are not proof of why a defect exists or of how the model 'reasons'.",
    "The final decision always rests with the inspector.",
]


class AppState:
    def __init__(self, app_cfg):
        self.app_cfg = app_cfg
        self.cfg = load_config(app_cfg["main_config"])
        self.names = self.cfg["class_names"]
        self.exif_mode = self.cfg.get("exif_mode", "apply")
        self.detector = build_detector(app_cfg["backend"], self.names, app_cfg.get("weights"), app_cfg.get("svm_bundle"))
        self.lock = threading.Lock()                     # one model call at a time
        self.display_floor = float(app_cfg.get("display_floor", 0.15))
        self.limits = app_cfg["limits"]
        self.warnings = []
        tj = app_cfg.get("threshold_json")
        if tj and Path(tj).exists():
            meta = json.load(open(tj))
            self.thr, self.thr_meta, self.thr_source = float(meta["conf_thr"]), meta, "validation"
        else:
            self.thr, self.thr_meta, self.thr_source = 0.5, {}, "DEFAULT (not validated)"
            self.warnings.append("Decision threshold is a default, not chosen on validation data. Run select_threshold.py.")
        cj = app_cfg.get("calibration_json")
        self.calib = json.load(open(cj)) if cj and Path(cj).exists() else None
        if self.calib is None:
            self.warnings.append("No validation reliability table found: scores are shown without 'seen in validation' figures.")
        oj = app_cfg.get("ood_json")
        self.ood = json.load(open(oj)) if oj and Path(oj).exists() else None
        if self.ood is None:
            self.warnings.append("No out-of-distribution statistics found: unusual inputs will not be flagged. Run fit_ood.py.")
        self.limitations = LIMITATIONS + list(app_cfg.get("extra_limitations") or [])

    def calibration_for(self, conf):
        if not self.calib:
            return None
        for b in self.calib["bins"]:
            if b["lo"] <= conf < b["hi"] or (b["hi"] >= 1 and conf <= b["hi"] and conf >= b["lo"]):
                if b["precision"] is None or b["n"] < int(self.app_cfg.get("min_validation_n", 20)):
                    return {"available": False, "n": b["n"]}
                return {"available": True, "precision": b["precision"], "ci": [b["ci_lo"], b["ci_hi"]], "n": b["n"]}
        return None

    def info(self):
        return {"model_version": self.app_cfg.get("model_version", ""), "backend": self.detector.name, "classes": {str(k): v for k, v in self.names.items()},
                "threshold": self.thr, "threshold_source": self.thr_source,
                "threshold_validation": {k: self.thr_meta.get(k) for k in ("target_recall", "any_defect_recall", "false_alarm_rate_on_defect_free_images")},
                "limitations": self.limitations, "limits": self.limits, "warnings": self.warnings,
                "explanations": {"occlusion": True, "eigencam": bool(self.detector.supports_eigencam)}}

    def predict(self, img):
        with self.lock:
            dets = self.detector.predict(img, self.display_floor)
        H, W = img.shape[:2]
        out = []
        for r in dets.sort_values("conf", ascending=False).itertuples():
            out.append({"cls": int(r.cls), "name": self.names.get(int(r.cls), str(int(r.cls))), "score": round(float(r.conf), 4),
                        "above_threshold": bool(r.conf >= self.thr), "box_norm": [float(r.x1), float(r.y1), float(r.x2), float(r.y2)],
                        "box_px": [int(r.x1 * W), int(r.y1 * H), int(r.x2 * W), int(r.y2 * H)], "validation": self.calibration_for(float(r.conf))})
        ood = ood_mod.score(img, self.ood) if self.ood else None
        flagged = [d for d in out if d["above_threshold"]]
        if ood and ood["out_of_distribution"]:
            verdict = {"decision": "REFER", "reason": "unusual_input",
                       "message": "This image looks unlike the fabric photos the model was trained on. Send it for manual inspection."}
        elif flagged:
            verdict = {"decision": "REFER", "reason": "defects_found",
                       "message": f"{len(flagged)} possible defect(s) found. Send for manual inspection."}
        else:
            verdict = {"decision": "PASS", "reason": "nothing_above_threshold",
                       "message": "No defect scored above the decision threshold. This does not guarantee the fabric is defect-free."}
        return out, verdict, ood
