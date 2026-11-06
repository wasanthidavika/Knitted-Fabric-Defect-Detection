#!/usr/bin/env bash
# Q4/Q5 smoke test on synthetic data. Run from the project root AFTER tests/run_q3_smoke.sh (needs its resized data and predictions).
set -e
export ISL_CONFIG=configs/config_smoke.yaml
python tests/test_xai_core.py
python tests/test_app.py
test -f smoke_out/outputs/q3/preds/SVM_BEST_val.csv || { echo "run tests/run_q3_smoke.sh first"; exit 1; }
python src/fit_ood.py
python src/calibration_check.py --preds smoke_out/outputs/q3/preds/SVM_BEST_val.csv --name SVM_BEST > /dev/null
python src/xai_validate.py --backend mock --split val --n 6 --cases --tag smoke --max-images 40 \
       --threshold-json smoke_out/outputs/q3/eval/threshold_SVM_BEST.json
echo "Q4/Q5 SMOKE TEST FINISHED"
