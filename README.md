# ISL-Knit fabric defect detection: Q1 and Q2 code

Free-Kaggle-notebook workflow: the 15 GB dataset is never downloaded to your laptop and nothing paid is used.

## File structure
```
isl_knit_project/
  README.md
  requirements.txt
  configs/
    config.yaml            Kaggle paths, seed 42, class names, expected split counts
    config_smoke.yaml      paths for the synthetic smoke test
  src/
    common.py              config loading, seeds, YOLO label reader
    audit_isl_knit.py      leakage audit + frozen group-aware split (proposed_splits.csv)
    prepare_data.py        resize to 640, EXIF handling, YOLO folders, data.yaml / data_test.yaml, label check
    verify_split.py        proves no group crosses partitions; compares counts with the addendum
    eda.py                 Q1 tables, figures, auto findings.md
    colour_shortcut.py     Q1 colour-confound check (train -> val only)
    model_summary.py       Q2.a layers, shapes, parameters, activations, GFLOPs
    make_arch_diagram.py   Q2.a architecture diagram from the real layer table
  docs/
    requirements_q1_q2.md  every Q1/Q2 bullet of the brief -> where it is covered
    q1_report_template.md  report skeleton for Q1
    q2_report_template.md  report skeleton for Q2 (decision table, alternatives, prediction)
  tests/
    make_synthetic.py      fake dataset (EXIF rotation, planted duplicates, imbalance)
    run_smoke.sh           runs the whole Q1 pipeline on fake data
    fixtures/              hand-made layer table used only to test the diagram script
  logs/experiments.csv     experiment log (header only; used from Q3 onwards)
  notebooks/               keep thin notebooks that only call src/*.py
```

## Run order on Kaggle
1. Upload this folder as a private Kaggle Dataset (Create -> New Dataset), then in a new notebook use Add Input to attach both it and ISL-Knit. Copy the code: `!cp -r /kaggle/input/<your-dataset>/isl_knit_project /kaggle/working/ && cd /kaggle/working/isl_knit_project`.
2. Settings: Accelerator None for Q1 (CPU is enough); `!pip -q install ultralytics` if it is missing. Check `raw_root` in `configs/config.yaml` with `!ls /kaggle/input/*`.
3. Run, in order:
```
!python src/audit_isl_knit.py --root <raw_root> --out /kaggle/working/audit
!python src/prepare_data.py --check          # then OPEN outputs/q1/label_check.png
!python src/verify_split.py                  # must print OK and match 2360 / 502 / 513
!python src/eda.py --examples
!python src/colour_shortcut.py
!python src/model_summary.py                 # Q2.a
!python src/make_arch_diagram.py             # Q2.a (needs layers.csv from the previous step)
```
4. Save Version, then download only small files: `outputs/` (figures, tables, markdown) and `data/yolo640/data.yaml`. Save `/kaggle/working/data/yolo640` as a private Kaggle Dataset so later notebooks reuse it.

## Smoke test (no real data needed, any machine)
```
pip install -r requirements.txt
bash tests/run_smoke.sh
```

## Rules baked into the code
- Test split is never used for analysis: `eda.py` and `colour_shortcut.py` use train+val only; `data.yaml` has no test entry; use `data_test.yaml` only in the final evaluation script.
- `prepare_data.py` clears `images/` and `labels/` before writing, so stale files from an older split cannot leak in.
- If `label_check.png` shows misaligned boxes on EXIF-rotated images, set `exif_mode: ignore` and re-run `prepare_data.py`.

## Tested vs untested
Tested (synthetic data): audit, prepare_data, verify_split, eda, colour_shortcut, make_arch_diagram (on a fixture layer table).
NOT tested: `model_summary.py` (needs torch + ultralytics, which would not install in the authoring sandbox), and nothing has run on the real ISL-Knit files yet. Expect small fixes on first real run.
