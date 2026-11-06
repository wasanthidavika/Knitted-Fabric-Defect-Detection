# Brief requirements for Q3 -> where each is covered
Source: AI_Coursework_2025_Batch_1.pdf, Question 3 (20 marks) and the rules in sections 5, 6 and 12.

## Q3.a Development, baseline comparison, error analysis (10 marks)
| Brief bullet | Covered by | You still write |
|---|---|---|
| Describe the initial neural model and how it was improved | `train_yolo.py` (E00 = initial), `configs/q3_plan.csv` (each improvement is a logged run), `run_experiments.py` | The narrative: what E00 is and which changes helped, with log ids |
| Present and interpret learning curves (under/over-fitting or otherwise limited) | `learning_curves.py` -> `figures/learning_curve_<run>.png`, `tables/learning_curve_diagnosis.csv` | The interpretation, using evidence (see template section 3) |
| Present the classical baseline, its search space and number of configurations | `baseline_lib.py`, `baseline_train.py` -> `baseline/baseline_summary.json`, `grid_results.csv` | Describe the design, search space (C x gamma = 9) + NMS setting (3) = 12 configurations |
| Comparison table: trivial benchmark, classical baseline, initial ANN, final ANN; primary metric, training time, inference latency | `final_eval.py` -> `final_comparison.csv/.md` | Paste the table; state hardware and what the timings include |
| Error analysis on the final model by class, subgroup, input type, data-quality issue | `error_analysis.py` -> confusion matrix, per class, by size, by colour, by crowding, worst cases | The analysis (prompts are in `summary.md`) |

## Q3.b Optimisation study (10 marks)
| Brief bullet | Covered by | You still write |
|---|---|---|
| Strengths/weaknesses of optimisation techniques, compared with your own evidence | runs E00 / E01 / E02 / E03 (+ curves) | Discussion tied to curves (template section 6) |
| At least THREE hyperparameters investigated with results | learning rate (E02, E03), batch size (E04, E05), depth/width (E06, E07), plus risky runs E08-E11 | Tables + conclusions supported by numbers |
| Compare at least TWO optimisers empirically, relate to mechanics | E00 (AdamW) vs E01 (SGD + momentum) | Explain behaviour using the curves |
| At least TWO substantive approaches that did not work, diagnosed | E08-E11 are designed to be informative if they fail; `auto_flags` helps diagnose | Diagnosis with evidence. Do NOT invent failures; report what happened |
| Complete experiment log consistent with the report | `logs/experiments.csv` written by `run_experiments.py` / `baseline_train.py` | `what_was_learned` for EVERY row |

## Rules from the brief enforced by the code
- Same split everywhere, test only once: `guard_test`, `final_eval.py --confirm`, lock file `outputs/q3/TEST_EVALUATED.lock`.
- Test error analysis only after the final evaluation, from saved predictions (no second evaluation).
- Thresholds chosen on validation only: `select_threshold.py`, passed to the later scripts with `--threshold-json`.
- Comparable tuning effort: ANN 12 planned configurations vs classical 12 (9 SVM + 3 NMS). State both numbers.
- Fixed seeds: 42 in the plan; final model seeds 42/43/44 (`configs/q3_final_plan.csv`); report mean and spread if results vary.
- Honest reporting: if the SVM wins, say so. Do not weaken the baseline.
- Declare AI-tool use in your appendix.
