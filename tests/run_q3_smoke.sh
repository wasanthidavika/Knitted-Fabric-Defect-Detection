#!/usr/bin/env bash
# Q3 smoke test on synthetic data (run from the project root). Needs the Q1 scripts (audit, prepare_data) in src/.
set -e
export ISL_CONFIG=configs/config_smoke.yaml
rm -rf smoke_out smoke_data
python tests/unit_metrics.py
python tests/make_synthetic_q3.py
python src/audit_isl_knit.py --root smoke_data/Full --out smoke_out/audit > /dev/null
python src/prepare_data.py --check
python src/verify_split.py > /dev/null && echo "verify_split OK"
# --- classical baseline ---
python src/baseline_train.py --stage fit_stage1 --stage1-images 60 --K 120
python src/baseline_train.py --stage cache --split train
python src/baseline_train.py --stage cache --split val
python src/baseline_train.py --stage grid --quick
python src/baseline_predict.py --split val --name SVM_BEST
P=smoke_out/outputs/q3/preds/SVM_BEST_val.csv
python src/select_threshold.py --preds $P --name SVM_BEST
T=smoke_out/outputs/q3/eval/threshold_SVM_BEST.json
python src/evaluate_detections.py --preds $P --split val --name SVM_BEST --threshold-json $T > smoke_out/eval_print.txt
python src/error_analysis.py --preds $P --split val --name SVM_BEST --threshold-json $T > smoke_out/err_print.txt
# --- experiment runner + learning curves with the stub trainer ---
python src/run_experiments.py --plan configs/q3_plan.csv --train-script tests/stub_train_yolo.py --only E00 E03 E10 > /dev/null
python src/learning_curves.py --runs E00 E03 E10 --compare smoke
cat smoke_out/experiments.csv | cut -c1-230
# --- final evaluation guard ---
python src/final_eval.py --baseline smoke_out/outputs/q3/baseline/best.joblib --skip-yolo && echo "SHOULD HAVE FAILED (no --confirm)" && exit 1 || echo "guard OK: refused without --confirm"
python src/error_analysis.py --preds $P --split test --name X && echo "SHOULD HAVE FAILED" && exit 1 || echo "guard OK: test error analysis refused before final_eval"
python src/final_eval.py --confirm --baseline smoke_out/outputs/q3/baseline/best.joblib --skip-yolo
python src/final_eval.py --confirm --baseline smoke_out/outputs/q3/baseline/best.joblib --skip-yolo && echo "SHOULD HAVE FAILED (lock)" && exit 1 || echo "guard OK: second test evaluation refused"
python src/error_analysis.py --preds smoke_out/outputs/q3/preds/classical_SVM_baseline_test.csv --split test --name SVM_BEST > smoke_out/err_test_print.txt
echo "Q3 SMOKE TEST FINISHED"
