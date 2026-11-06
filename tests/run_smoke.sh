#!/usr/bin/env bash
# Smoke test of the Q1 pipeline on synthetic data (run from the project root).
set -e
export ISL_CONFIG=configs/config_smoke.yaml
rm -rf smoke_out
python tests/make_synthetic.py
python src/audit_isl_knit.py --root smoke_data/Full --out smoke_out/audit
python src/prepare_data.py --check
python src/verify_split.py
python src/eda.py --examples
python src/colour_shortcut.py
# Q2 (needs torch + ultralytics):
python src/model_summary.py && python src/make_arch_diagram.py || echo "Q2 scripts skipped/failed (torch or ultralytics missing?)"
echo "SMOKE TEST FINISHED"
