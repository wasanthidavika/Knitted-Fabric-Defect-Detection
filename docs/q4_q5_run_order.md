# Q4 + Q5: file placement and run order

## File placement (keep folder names)
| File | Goes to |
|---|---|
| `app/` (main.py, inference.py, validation.py, app_config.yaml, static/index.html, app.js, style.css, `__init__.py`) | `app/` (new folder at the project root) |
| `src/detectors.py, xai_core.py, ood.py, fit_ood.py, calibration_check.py, xai_validate.py` | `src/` |
| `tests/test_app.py, test_xai_core.py, run_q45_smoke.sh` | `tests/` |
| `docs/*.md` | `docs/` |
| `requirements_q45.txt` | project root |
Nothing from Q1-Q3 is modified. Uses `error_analysis.match`, `q3_utils`, `metrics` from Q3.

## 0. Test without a model (any machine)
```
pip install -r requirements_q45.txt
python tests/test_xai_core.py
python tests/test_app.py          # 31 API checks with a mock detector
```

## 1. On Kaggle after the final model exists (frozen, chosen on validation)
```
python src/yolo_predict.py --weights outputs/q3/F42/weights/best.pt --split val --name F42
python src/select_threshold.py --preds outputs/q3/preds/F42_val.csv --name F42 --target-recall 0.95
python src/calibration_check.py --preds outputs/q3/preds/F42_val.csv --name F42
python src/fit_ood.py
# choose the EigenCAM layer on VALIDATION (compare the deletion summaries), then the final run with cases:
python src/xai_validate.py --backend yolo --weights outputs/q3/F42/weights/best.pt --split val --preds outputs/q3/preds/F42_val.csv \
       --threshold-json outputs/q3/eval/threshold_F42.json --layer 15 --n 30 --tag L15
python src/xai_validate.py ... --layer 18 --tag L18        # and 21
python src/xai_validate.py ... --layer <chosen> --n 30 --cases --randomise --tag final
```
If the YOLO hook fails (`activation hook did not fire` or a shape error), send me the traceback: `YoloDetector.activation` and `--randomise` could not be run against a real Ultralytics install while writing this.

## 2. On your laptop (the app; nothing paid, no dataset needed)
Copy from Kaggle's outputs: `best.pt`, `threshold_F42.json`, `calibration_F42.json`, `ood_stats.json`, and `configs/config.yaml` (class names, exif_mode). Put them at the paths in `app/app_config.yaml` (edit the paths/model_version), then from the project root:
```
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000. The page warns if the threshold, reliability table or OOD statistics are missing.
CPU note: an explanation = about 40 model passes for occlusion + about 6 for each check; measure it on your laptop and lower `explain_max_patches` if it is slow.

## 3. Evidence to collect for the report and video
- Video (max 8 min, brief section 8) must show: a real-time prediction, the explanation feature, and at least one incorrect prediction. Suggested: 1) upload a defective image -> REFER; 2) "Why?" on a detection and read the sanity sentence; 3) upload a defect-free image that the model flags (false positive) or a defect it misses and say what the explanation suggests; 4) upload an invalid file and an unusual photo to show the error/warning; 5) show the limitations panel.
- Manual test table (docs/q4_report_template.md section 6) with screenshots, using the REAL model.
- `outputs/q5/deletion_summary_*.json`, `outputs/q5/cases/`, `randomisation_*.csv`, `outputs/q4/figures/reliability_*.png`.
- `logs/experiments.csv` rows with family "xai": write `what_was_learned`.

## Known limits (read before trusting)
- Tested with a mock detector and the SVM backend; the YOLO paths (`YoloDetector`, EigenCAM hooks, `--randomise`) are written against the Ultralytics API but NOT run here.
- Ultralytics treats numpy arrays as BGR: `YoloDetector` converts from RGB. Check one prediction in the app against `yolo_predict.py` output for the same image (they should agree closely).
- The OOD check uses 8 global statistics: a coarse safeguard, not a guarantee.
- The "box control" can beat the occlusion window; that is a finding, not a bug.
