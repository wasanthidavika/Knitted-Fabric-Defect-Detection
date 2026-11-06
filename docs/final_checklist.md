# Final student checklist (brief section 13) -> evidence you must be able to point to
Tick only when the evidence file exists and you can explain it.
| ☐ | Checklist item | Evidence (file / section) | Produced by |
|---|---|---|---|
| ☐ | Problem has an identified user and a stated decision | Q1 report section; addendum; app text | docs/q1_report_template.md |
| ☐ | Dataset and task approved by the lecturer | his approval email + your addendum (note: task moved from classification to detection, say so) | email |
| ☐ | One fixed train/val/test split throughout | `proposed_splits.csv`, `verify_split.py` output, same split for SVM, all YOLO runs, ViT, CLIP | Q1/Q3 |
| ☐ | Leakage prevented and documented | `audit/report.json`, `split_verification.json`, label/EXIF check, colour shortcut | Q1 |
| ☐ | Primary metric justified | Q1 report (mAP@0.5, costs of FN/FP) | Q1 |
| ☐ | ANN and classical baseline, comparable tuning effort | `baseline_summary.json` (12 configs) vs `q3_plan.csv` (12 runs); numbers stated in report | Q3 |
| ☐ | All substantive experiments recorded, including failures | `logs/experiments.csv` (yolo, svm, xai rows) with `what_was_learned` filled for EVERY row; every report number traceable to a row | Q3/Q5 |
| ☐ | Final models evaluated only once on the held-out test set | `outputs/q3/TEST_EVALUATED.lock` (YOLO, SVM) AND `outputs/q6/TEST_EVALUATED_Q6.lock` (ViT, CLIP); run both back to back with everything frozen | `final_eval.py`, `q6_final_eval.py` |
| ☐ | Error analysis conducted | `outputs/q3/error_analysis/...` | Q3 |
| ☐ | Web app generates predictions at request time | `app/`, video moment, test "prediction changes with the image" | Q4 |
| ☐ | XAI method appropriate and sanity-checked | `outputs/q5/deletion_summary_*.json`, randomisation CSV, layer/fill comparison | Q5 |
| ☐ | At least THREE incorrect predictions analysed with XAI | `outputs/q5/cases/cases.csv` (>= 3 rows not "correct") + your written analysis | Q5 |
| ☐ | Q6 completed with a meaningful GenAI application (or prior approval for an adjacent task) | `outputs/q6/q6_comparison.md`, `q6_paired_bootstrap.csv`, `clip_selection.csv` (all prompt combinations, val only), `vit_facts.md`, ViT runs V00-V03 in the log, `label_audit*.csv/png` (if used as the adjacent task: ask the lecturer first) | Q6 (`docs/q6_report_template.md`) |
| ☐ | Code, environment info, README, dataset/licence info, experiment log included | repo, `requirements*.txt` (pin versions with `pip freeze`), README with Kaggle reproduction steps, download script + sample (dataset > 500 MB), licence text | Task 7 |
| ☐ | Demo video <= 8 minutes with the required features | real-time prediction, explanation, at least one incorrect prediction (see run order) | Task 7 |
| ☐ | Report <= 25 pages excluding appendices, follows Q1-Q6 | page budget: Q1 4, Q2 5, Q3 6, Q4 2, Q5 4, Q6 4 | Task 7 |
| ☐ | Generative AI use declared | appendix: list every use (coding, debugging, analysis, writing, brainstorming) of Claude/other tools, incl. audit script, Q1-Q5 code, addendum, templates | Task 7 |
| ☐ | I can explain and defend every part | read every script; practise small live changes (threshold, class list, layer, patch size) | you |

## Notes from the brief that are easy to miss
- Experiment log: "Any result discussed in the report must be traceable to the log." Mark every figure in the report with its log id (E03, S05, XAI_final_eccam...).
- Seeds: report 42/43/44 for the final model; mean and spread if results vary.
- Test set: thresholds, prompts, calibration, OOD limit, EigenCAM layer and all other decisions come from train/val only.
- If the classical baseline wins, report it without modification and explain with evidence.
