"""
API tests for the web app (Q4 test table): normal, invalid and out-of-distribution inputs, explanation endpoint.
Uses the MOCK detector so it runs anywhere (no GPU / trained model). Run from the project root:  python tests/test_app.py
Optional: if the Q3 smoke test left an SVM bundle, the SVM backend is exercised too.
"""
import io, json, os, sys, tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import ood

tmp = Path(tempfile.mkdtemp())


def fabric(seed=0, defect=True, size=(320, 240)):
    r = np.random.default_rng(seed)
    W, H = size
    a = np.full((H, W, 3), 140.0) + r.normal(0, 6, (H, W, 3))
    if defect:
        a[100:130, 180:215] -= 70
    return a.clip(0, 255).astype(np.uint8)


def jpeg(arr, exif=None):
    b = io.BytesIO()
    im = Image.fromarray(arr)
    if exif:
        e = Image.Exif(); e[274] = exif; im.save(b, "JPEG", exif=e)
    else:
        im.save(b, "JPEG", quality=92)
    return b.getvalue()


# --- test configuration: mock model, validation-style artefacts written to a temp folder
def via_jpeg(a):
    return np.asarray(Image.open(io.BytesIO(jpeg(a))).convert("RGB"))
stats = ood.fit(np.stack([ood.features(via_jpeg(fabric(s, defect=bool(s % 2)))) for s in range(60)]))
json.dump(stats, open(tmp / "ood.json", "w"))
json.dump({"conf_thr": 0.6, "target_recall": 0.95, "any_defect_recall": 0.96, "false_alarm_rate_on_defect_free_images": 0.2}, open(tmp / "thr.json", "w"))
bins = [{"lo": round(i / 10, 1), "hi": round((i + 1) / 10, 1), "n": 50 if i >= 5 else 5, "n_correct": 30, "precision": 0.6, "ci_lo": 0.46, "ci_hi": 0.72} for i in range(10)]
json.dump({"bins": bins}, open(tmp / "cal.json", "w"))
cfg = {"main_config": str(ROOT / "configs" / "config_smoke.yaml") if (ROOT / "configs" / "config_smoke.yaml").exists() else "configs/config_smoke.yaml",
       "backend": "mock", "threshold_json": str(tmp / "thr.json"), "calibration_json": str(tmp / "cal.json"), "ood_json": str(tmp / "ood.json"),
       "model_version": "test", "work_long_side": 320, "display_floor": 0.15, "min_validation_n": 20, "explain_max_patches": 25,
       "limits": {"max_upload_mb": 1, "min_side_px": 64, "max_side_px": 12000, "max_pixels": 120000000, "max_aspect_ratio": 8}}
yaml.safe_dump(cfg, open(tmp / "app.yaml", "w"))
os.environ["ISL_APP_CONFIG"] = str(tmp / "app.yaml")

from fastapi.testclient import TestClient
from app.main import app

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))

with TestClient(app, raise_server_exceptions=False) as c:
    r = c.get("/health"); check("health", r.status_code == 200 and r.json()["backend"] == "mock")
    r = c.get("/"); check("frontend served", r.status_code == 200 and "Knitted fabric defect check" in r.text)
    info = c.get("/api/info").json()
    check("info: classes, limitations, threshold", len(info["classes"]) == 7 and len(info["limitations"]) >= 5 and info["threshold"] == 0.6 and info["threshold_source"] == "validation")
    check("info: eigencam flagged unavailable for mock", info["explanations"]["eigencam"] is False)

    # ---- NORMAL inputs, and proof that predictions are computed per request
    d = c.post("/api/predict", files={"file": ("a.jpg", jpeg(fabric()), "image/jpeg")}); j = d.json()
    check("normal defective image -> 200 REFER", d.status_code == 200 and j["verdict"]["decision"] == "REFER" and j["verdict"]["reason"] == "defects_found", str(j.get("verdict")))
    det = j["detections"][0] if j.get("detections") else {}
    check("detection has score, box, validation info", det.get("score", 0) >= 0.6 and len(det.get("box_norm", [])) == 4 and det.get("validation", {}).get("available"))
    check("response has preview, timing, ood block", len(j.get("preview_jpeg", "")) > 100 and j.get("inference_ms") is not None and j.get("ood") is not None)
    d2 = c.post("/api/predict", files={"file": ("b.jpg", jpeg(fabric(defect=False)), "image/jpeg")}); j2 = d2.json()
    check("normal clean image -> 200 PASS", d2.status_code == 200 and j2["verdict"]["decision"] == "PASS", str(j2.get("verdict")))
    check("prediction changes with the image (computed at request time)", j["detections"] != j2["detections"])
    j3 = c.post("/api/predict", files={"file": ("c.jpg", jpeg(fabric(seed=5)), "image/jpeg")}).json()
    check("PASS message does not claim defect-free", "does not guarantee" in j2["verdict"]["message"])

    # ---- EXIF handling matches training preprocessing
    jx = c.post("/api/predict", files={"file": ("x.jpg", jpeg(fabric(), exif=6), "image/jpeg")}).json()
    check("EXIF rotation applied (size swapped)", jx["image"]["exif_rotation_applied"] and jx["image"]["work_size"][1] > jx["image"]["work_size"][0], str(jx.get("image")))

    # ---- INVALID inputs
    def status(files): return c.post("/api/predict", files=files).status_code
    check("empty file -> 400", status({"file": ("e.jpg", b"", "image/jpeg")}) == 400)
    check("text file renamed .jpg -> 415", status({"file": ("t.jpg", b"hello this is not an image", "image/jpeg")}) == 415)
    check("truncated JPEG -> 415", status({"file": ("tr.jpg", jpeg(fabric())[:300], "image/jpeg")}) == 415)
    check("tiny image -> 422", status({"file": ("s.jpg", jpeg(fabric(size=(32, 32))), "image/jpeg")}) == 422)
    check("extreme aspect ratio -> 422", status({"file": ("w.jpg", jpeg(np.full((100, 2000, 3), 120, np.uint8)), "image/jpeg")}) == 422)
    buf = io.BytesIO(); Image.fromarray((np.random.default_rng(0).random((900, 900, 3)) * 255).astype(np.uint8)).save(buf, "PNG")
    check("upload larger than limit -> 413", status({"file": ("big.png", buf.getvalue(), "image/png")}) == 413, str(len(buf.getvalue())))
    big = io.BytesIO(); Image.fromarray(np.full((1500, 13000, 3), 128, np.uint8)).save(big, "PNG")
    check("dimensions above limit -> 413", status({"file": ("huge.png", big.getvalue(), "image/png")}) == 413)
    check("missing file field -> 422", c.post("/api/predict").status_code == 422)
    check("error body is plain language (no stack trace)", "Traceback" not in c.post("/api/predict", files={"file": ("t.jpg", b"xx", "image/jpeg")}).text)

    # ---- OUT-OF-DISTRIBUTION inputs
    for name, arr in (("blank white", np.full((240, 320, 3), 255, np.uint8)), ("random noise", (np.random.default_rng(3).random((240, 320, 3)) * 255).astype(np.uint8)),
                      ("colour gradient", np.tile(np.linspace(0, 255, 320).astype(np.uint8)[None, :, None], (240, 1, 3)))):
        jo = c.post("/api/predict", files={"file": ("o.jpg", jpeg(arr), "image/jpeg")}).json()
        check(f"OOD '{name}' -> REFER (unusual input)", jo["verdict"]["reason"] == "unusual_input" and jo["ood"]["out_of_distribution"], str(jo.get("verdict")))
    check("normal fabric NOT flagged as OOD", j["ood"]["out_of_distribution"] is False, str(j["ood"]))

    # ---- EXPLAIN endpoint
    bx = det["box_norm"]
    form = {"x1": bx[0], "y1": bx[1], "x2": bx[2], "y2": bx[3], "cls": det["cls"], "methods": "occlusion,eigencam"}
    e = c.post("/api/explain", files={"file": ("a.jpg", jpeg(fabric()), "image/jpeg")}, data=form); ej = e.json()
    occ = ej.get("methods", {}).get("occlusion", {})
    check("explain: occlusion overlay + sanity numbers", e.status_code == 200 and occ.get("available") and len(occ.get("overlay_png", "")) > 100 and "drop_when_hiding_highlight" in occ, str(ej)[:200])
    check("explain: hiding highlight lowers score more than random", occ.get("drop_when_hiding_highlight", 0) > occ.get("drop_when_hiding_random", 1) + 0.05, str(occ.get("drop_when_hiding_highlight")))
    check("explain: eigencam reported unavailable gracefully", ej["methods"]["eigencam"]["available"] is False and "YOLO" in ej["methods"]["eigencam"]["reason"])
    bad = dict(form, x1=0.9, x2=0.1)
    check("explain: invalid box -> 422", c.post("/api/explain", files={"file": ("a.jpg", jpeg(fabric()), "image/jpeg")}, data=bad).status_code == 422)
    check("explain: unknown class -> 422", c.post("/api/explain", files={"file": ("a.jpg", jpeg(fabric()), "image/jpeg")}, data=dict(form, cls=99)).status_code == 422)
    miss = dict(form, x1=0.05, y1=0.05, x2=0.15, y2=0.15)
    em = c.post("/api/explain", files={"file": ("a.jpg", jpeg(fabric()), "image/jpeg")}, data=miss).json()
    check("explain: box with no detection -> clear note, no crash", em.get("note") and em["methods"]["occlusion"]["available"] is False)

# ---- optional: real SVM backend through the same code path
bundle = ROOT / "smoke_out" / "outputs" / "q3" / "baseline" / "best.joblib"
if bundle.exists():
    from detectors import SvmDetector
    det = SvmDetector(bundle, {i: str(i) for i in range(7)})
    out = det.predict(fabric(size=(480, 360)), 0.05)
    check("SVM backend returns well-formed detections", list(out.columns) == ["cls", "conf", "x1", "y1", "x2", "y2"] and ((out.conf >= 0) & (out.conf <= 1)).all())
else:
    print("(SVM backend test skipped: run tests/run_q3_smoke.sh first)")

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} app tests passed")
sys.exit(1 if failed else 0)
