#!/usr/bin/env bash
# Q6 smoke test on synthetic data with a FAKE CLIP scorer (pipeline only). Run from the project root AFTER tests/run_q3_smoke.sh.
set -e
export ISL_CONFIG=configs/config_smoke.yaml
export Q6_FAKE=1
python tests/test_image_level.py
python tests/test_clip_pipeline.py
test -f smoke_out/outputs/q3/preds/classical_SVM_baseline_test.csv || { echo "run tests/run_q3_smoke.sh first"; exit 1; }
rm -rf smoke_out/outputs/q6
python src/vit_facts.py > /dev/null && echo "vit_facts OK"
python src/clip_zeroshot.py --stage select > smoke_out/clip_select.txt && echo "clip select OK"
python src/clip_zeroshot.py --stage scores --split val > /dev/null && echo "clip val scores OK"
python src/clip_zeroshot.py --stage latency > /dev/null && echo "clip latency OK"
python src/clip_label_audit.py > smoke_out/audit_print.txt && echo "label audit OK"
COMMON="--svm-val smoke_out/outputs/q3/preds/SVM_BEST_val.csv --svm-test smoke_out/outputs/q3/preds/classical_SVM_baseline_test.csv --skip-vit --clip"
python src/q6_final_eval.py $COMMON && echo "SHOULD HAVE FAILED (no --confirm)" && exit 1 || echo "guard OK: refused without --confirm"
python src/q6_final_eval.py --confirm $COMMON
python src/q6_final_eval.py --confirm $COMMON && echo "SHOULD HAVE FAILED (lock)" && exit 1 || echo "guard OK: second Q6 test evaluation refused"
echo "Q6 SMOKE TEST FINISHED"
