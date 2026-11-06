# Brief requirements for Q1 and Q2 -> where each is covered

Source: AI_Coursework_2025_Batch_1.pdf (sections 4, 5, 6, 9, 12, 13). "Manual" = you must write it; the code gives you the evidence.

## Q1 - Dataset and EDA (10 marks)
| Brief bullet | Covered by | Status |
|---|---|---|
| Describe dataset: source, licence/permission, structure, size, target, characteristics | `outputs/q1/tables/dataset_description.csv`, `findings.md` (summary), Kaggle page + Heliyon paper | Manual: copy licence wording from the Kaggle page; cite the paper |
| Comprehensive EDA appropriate to the modality | `src/eda.py`: class balance, boxes/image, box size, position heatmap, co-occurrence, image sizes, colour, `--examples` crops | Code ready |
| Costs of false positives vs false negatives | `docs/q1_report_template.md` section 3 | Manual (text drafted in addendum) |
| Define and justify the primary metric | template section 4 | Manual |
| Leakage checks and data-quality handling | `audit_isl_knit.py` (duplicates, groups), `verify_split.py`, `prepare_data.py --check` (labels/EXIF), `colour_shortcut.py` | Code ready |
| End with three findings that constrain modelling | `outputs/q1/findings.md` (auto numbers + consequences) | Code ready; review wording |
| Sec. 4: document transformations that affect model input | resize long side 640, EXIF handling, JPEG q95 (see `prepare_data.py`, `manifest.csv`) | Describe in report |
| Sec. 4: group leakage (rolls/lots) | No roll/lot IDs exist; limitation stated in addendum | Manual: state the limitation |

## Q2.a - Architecture (10 marks)
| Brief bullet | Covered by | Status |
|---|---|---|
| Layers, dimensions, activations, parameter count | `src/model_summary.py` -> `layers.csv`, `q2_architecture_table.md`, `param_summary.json` | Run on Kaggle (not tested here) |
| Clear architecture diagram | `src/make_arch_diagram.py` -> `architecture.png` / `.mmd` | Layout tested on a fixture; run after model_summary |
| Assumptions the architecture makes about the data | `docs/q2_report_template.md` section 2 | Manual |
| Justify depth, width, normalisation, regularisation, output layer, loss individually | template section 3 (decision table) | Manual; verify each default against `param_summary.json` and Ultralytics docs |
| Loss/output consistent with Q1 decision costs | template section 3 (last rows) | Manual |

## Q2.b - Alternatives (10 marks)
| Brief bullet | Covered by | Status |
|---|---|---|
| Meaningful alternatives and why the chosen approach | template section 4 table | Manual |
| Why an ANN fits modality, volume, structure, task | template section 4 | Manual |
| State BEFORE results which family wins and why | template section 5 (prediction already in addendum) | Keep unchanged; cite date |
| For each rejected alternative: when appropriate / why less suitable here | table columns | Manual |

## Rules from the brief that already apply
- One fixed split (sec. 6): `proposed_splits.csv`, checked by `verify_split.py`.
- Test-set isolation (sec. 6): EDA decisions use train+val only; `data.yaml` has no test split; `data_test.yaml` is for the final evaluation only.
- Fixed seeds (sec. 6): `seed: 42` in the config, applied in every script.
- Declare AI use (sec. 7): list the audit script, this code, the addendum and guides in your appendix; read every line before submitting.
