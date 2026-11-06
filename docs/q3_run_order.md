# Q3 run order on Kaggle (and where each file goes)

## File placement (copy into your project; keep folder names)
| File | Goes to | Note |
|---|---|---|
| src/metrics.py, q3_utils.py, run_experiments.py, learning_curves.py, yolo_predict.py, evaluate_detections.py, error_analysis.py, select_threshold.py, final_eval.py, baseline_lib.py, baseline_train.py, baseline_predict.py | `src/` | new |
| src/train_yolo.py | `src/` | REPLACES yours; only adds optional flags (diff it). Needed for E08, E11 (--freeze, --cls) and the momentum flag in E01 |
| configs/q3_plan.csv, configs/q3_final_plan.csv | `configs/` | new |
| docs/*.md | `docs/` | new |
| tests/* | `tests/` | new (smoke test without real data) |
Your own eda.py / config.yaml / common.py / prepare_data.py / verify_split.py are NOT touched. Optional config key: `experiment_log: logs/experiments.csv` (that is already the default).

## Before you start
- `pip install scikit-image` if missing (Gabor/GLCM/LBP); OpenCV, scikit-learn, joblib come with Kaggle.
- `python tests/unit_metrics.py` (instant) and `bash tests/run_q3_smoke.sh` (several minutes, synthetic data) prove the code runs on your machine.
- COMMIT `configs/q3_plan.csv` to git BEFORE running it: the commit hash in the log proves the hypotheses were written first.
- Make sure verify_split.py prints OK and label_check.png looks right (Q1/Q2 steps).

## Classical baseline (CPU session, accelerator None)
```
python src/baseline_train.py --stage fit_stage1 --workers 4
python src/baseline_train.py --stage cache --split val   --workers 4 --limit 20   # TIME 20 images first
python src/baseline_train.py --stage cache --split train --workers 4
python src/baseline_train.py --stage cache --split val   --workers 4
python src/baseline_train.py --stage grid
python src/baseline_predict.py --split val --name SVM_BEST
```
Rough cost (extrapolated from a small sandbox CPU, NOT measured on Kaggle): about 1 second of CPU per 640 px image for features. Train cache over 2,360 images is probably tens of minutes to an hour+. Time `--limit 20` first. Caches are per image on disk, so a dead session resumes where it stopped (if you saved the working directory as a dataset / version).
If `cascade_recall_upper_bound_val` in `baseline_summary.json` is low (< ~0.8), the cascade, not the SVM, limits the baseline: raise `--K` (rebuild caches) before blaming the classifier.

## Neural runs (GPU session)
```
python src/run_experiments.py --plan configs/q3_plan.csv --dry-run          # check commands
python src/run_experiments.py --plan configs/q3_plan.csv --max-runs 1        # E00, TIME it
python src/run_experiments.py --plan configs/q3_plan.csv --time-budget-min 480
python src/learning_curves.py --runs E00 E01 E02 E03 E04 E05 E06 E07 E08 E09 E10 E11
python src/learning_curves.py --runs E00 E02 E03 --compare learning_rate
```
Time = runs x epochs x seconds per epoch. Patience is 10 in train_yolo.py, so some runs may stop early (flag `stopped_early`): note it when comparing training times.

## Pick the final configuration (validation only), then final training
Edit `configs/q3_final_plan.csv` (replace every TODO), commit it, then
`python src/run_experiments.py --plan configs/q3_final_plan.csv`.

## Predictions, thresholds and cross-checks on VALIDATION
```
python src/yolo_predict.py --weights outputs/q3/E00/weights/best.pt --split val --name E00
python src/evaluate_detections.py --preds outputs/q3/preds/E00_val.csv --split val --name E00 --compare-log-id E00
python src/select_threshold.py --preds outputs/q3/preds/E00_val.csv --name E00 --target-recall 0.95
python src/error_analysis.py --preds outputs/q3/preds/E00_val.csv --split val --name E00 --threshold-json outputs/q3/eval/threshold_E00.json
```
Repeat for the final run and for SVM_BEST. Use each model's own validation threshold (confidence scales differ: SVM confidences are a sigmoid of an SVM score margin, YOLO's are class scores).

## THE ONE test evaluation (everything frozen!)
```
python src/final_eval.py --confirm \
  --baseline outputs/q3/baseline/best.joblib \
  --yolo-initial outputs/q3/E00/weights/best.pt --initial-id E00 \
  --yolo-final outputs/q3/F42/weights/best.pt,outputs/q3/F43/weights/best.pt,outputs/q3/F44/weights/best.pt --final-ids F42,F43,F44 \
  --thr-baseline outputs/q3/eval/threshold_SVM_BEST.json --thr-initial outputs/q3/eval/threshold_E00.json --thr-final outputs/q3/eval/threshold_F42.json
python src/error_analysis.py --preds outputs/q3/preds/final_ANN_F42_test.csv --split test --name F42 --threshold-json outputs/q3/eval/threshold_F42.json
```
The lock file stops a second run. Do not retune after seeing test numbers.

## Known limits (read before trusting numbers)
- Tested only on synthetic data; `train_yolo.py`, `yolo_predict.py` and the Ultralytics paths could NOT be run in the authoring sandbox (no torch/ultralytics). Expect a small fix on first real run, and tell me the error.
- The new Ultralytics flags (freeze, cls, box, momentum, cos_lr, lrf, warmup_epochs) are standard train arguments to my knowledge: verify with your installed version.
- Our mAP (COCO-style 101-point) differs slightly from Ultralytics' own; report the gap.
- One seed per screening run: screening differences can be noise.
