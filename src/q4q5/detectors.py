"""
Detector interface shared by the web app (Q4) and the XAI code (Q5).

    predict(img_rgb_uint8, conf)  -> DataFrame[cls, conf, x1, y1, x2, y2]   (boxes normalised 0..1)
    predict_batch(list_of_imgs, conf)

Backends:
  yolo : the trained ANN (Ultralytics). Supports EigenCAM through forward hooks.
  svm  : the classical baseline (Q3), useful for comparison and for testing the app without a GPU
  mock : deterministic toy detector for automated tests ONLY (finds regions that differ strongly from the local background)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

COLS = ["cls", "conf", "x1", "y1", "x2", "y2"]


class Detector:
    name = "base"
    supports_eigencam = False
    imgsz = 640

    def __init__(self, class_names):
        self.class_names = {int(k): str(v) for k, v in class_names.items()}

    def predict(self, img_rgb, conf=0.001):
        raise NotImplementedError

    def predict_batch(self, imgs, conf=0.001):
        return [self.predict(i, conf) for i in imgs]

    def info(self):
        return {"backend": self.name}


class YoloDetector(Detector):
    name = "yolo"
    supports_eigencam = True

    def __init__(self, weights, class_names, imgsz=640, iou=0.7, max_det=300, device=None):
        super().__init__(class_names)
        from ultralytics import YOLO
        self.weights = str(weights)
        self.yolo = YOLO(self.weights)
        self.imgsz, self.iou, self.max_det, self.device = imgsz, iou, max_det, device

    def _run(self, imgs_rgb, conf):
        # Ultralytics treats numpy arrays as BGR (OpenCV convention): convert from RGB.
        bgr = [np.ascontiguousarray(i[..., ::-1]) for i in imgs_rgb]
        return self.yolo.predict(bgr, conf=conf, iou=self.iou, imgsz=self.imgsz, max_det=self.max_det,
                                 device=self.device, verbose=False)

    @staticmethod
    def _to_df(r):
        b = r.boxes
        if b is None or len(b) == 0:
            return pd.DataFrame(columns=COLS)
        return pd.DataFrame({"cls": b.cls.cpu().numpy().astype(int), "conf": b.conf.cpu().numpy().astype(float),
                             "x1": b.xyxyn[:, 0].cpu().numpy(), "y1": b.xyxyn[:, 1].cpu().numpy(),
                             "x2": b.xyxyn[:, 2].cpu().numpy(), "y2": b.xyxyn[:, 3].cpu().numpy()})

    def predict(self, img_rgb, conf=0.001):
        return self._to_df(self._run([img_rgb], conf)[0])

    def predict_batch(self, imgs, conf=0.001, chunk=16):
        out = []
        for i in range(0, len(imgs), chunk):
            out += [self._to_df(r) for r in self._run(imgs[i:i + chunk], conf)]
        return out

    def activation(self, img_rgb, layer_idx):
        """Feature map (C,h,w) of layer `layer_idx` (e.g. 15/18/21 = neck outputs at stride 8/16/32 for YOLOv8)
        and the padded network input size (H_in, W_in). UNTESTED against a real Ultralytics install: if the hook
        does not fire, check that `self.yolo.model.model` is the nn.Sequential of layers."""
        net = self.yolo.model.model
        store = {}
        h1 = net[layer_idx].register_forward_hook(lambda m, i, o: store.__setitem__("A", o.detach().float().cpu().numpy()[0]))
        h0 = net[0].register_forward_pre_hook(lambda m, i: store.__setitem__("in_hw", tuple(int(v) for v in i[0].shape[-2:])))
        try:
            self._run([img_rgb], 0.25)
        finally:
            h1.remove()
            h0.remove()
        if "A" not in store:
            raise RuntimeError("activation hook did not fire")
        return store["A"], store["in_hw"]

    def info(self):
        return {"backend": "yolo", "weights": Path(self.weights).name, "imgsz": self.imgsz, "nms_iou": self.iou}


class SvmDetector(Detector):
    name = "svm"

    def __init__(self, bundle_path, class_names):
        super().__init__(class_names)
        import joblib
        self.bundle_path = str(bundle_path)
        self.b = joblib.load(self.bundle_path)

    def predict(self, img_rgb, conf=0.001):
        from baseline_lib import extract_candidates, windows_to_detections
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        c = extract_candidates(gray, self.b["anchors"], self.b["stage1"], self.b["K"], self.b["nms_iou"])
        df = windows_to_detections(self.b, "x", c["boxes"], c["X"], c["hw"])
        df = df[df["conf"] >= conf]
        return df[COLS].reset_index(drop=True) if len(df) else pd.DataFrame(columns=COLS)

    def info(self):
        return {"backend": "svm", "bundle": Path(self.bundle_path).name}


class MockDetector(Detector):
    """TEST DOUBLE. Flags regions whose brightness differs strongly from a blurred background; confidence grows with
    contrast. Hiding such a region (occlusion) removes the detection, hiding elsewhere does not."""
    name = "mock"

    def __init__(self, class_names, min_contrast=25.0):
        super().__init__(class_names)
        self.min_contrast = min_contrast

    def predict(self, img_rgb, conf=0.001):
        g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        H, W = g.shape
        bg = cv2.GaussianBlur(g, (0, 0), max(H, W) / 12)
        diff = np.abs(g - bg)
        mask = (diff > self.min_contrast).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
        rows = []
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if a < 12:
                continue
            contrast = float(diff[lab == i].mean())
            c = float(np.clip(0.4 + contrast / 120.0, 0, 0.98))
            if c >= conf:
                rows.append((0, c, x / W, y / H, (x + w) / W, (y + h) / H))
        return pd.DataFrame(rows, columns=COLS)


def build_detector(backend, class_names, weights=None, bundle=None, device=None):
    if backend == "yolo":
        return YoloDetector(weights, class_names, device=device)
    if backend == "svm":
        return SvmDetector(bundle, class_names)
    if backend == "mock":
        return MockDetector(class_names)
    raise ValueError(f"unknown backend {backend!r}")
