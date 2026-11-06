# Brief requirements for Q6 -> where each is covered
Source: AI_Coursework_2025_Batch_1.pdf, Question 6 (20 marks), section 6 rules, and your lecturer's comment (a ViT treatment with patch tokens and sequence length).
"Manual" = you write it or do it with the real models (the sandbox could not run ViT/CLIP).

## Q6.a Apply a generative / foundation model (10 marks)
| Brief bullet | Covered by | Status |
|---|---|---|
| Apply a suitable pretrained foundation model to the approved problem where technically meaningful | ViT-B/16 fine-tuned (`vit_train.py`) and CLIP ViT-B/32 zero-shot (`clip_zeroshot.py`), both at image level | Code written; run on Kaggle (GPU) |
| Same held-out test examples and same primary metric wherever a meaningful comparison is possible | `q6_final_eval.py`: same test images, image-level macro-F1 with val-tuned thresholds for YOLO, SVM, ViT, CLIP (YOLO/SVM converted by max score per class) | Tested with SVM + fake CLIP |
| Why not the detection metric | mAP@0.5 needs boxes; ViT/CLIP give none, so the comparable metric is image-level. State this in the report | Manual |
| Prompts documented and chosen on VALIDATION only | `configs/q6_prompts.yaml` (all candidate prompts), `clip_selection.csv` (every prompt set x variant x aggregator, all logged), `clip_chosen.json`; test touched once in `q6_final_eval.py` | Ready; edit descriptions first |
| Adjacent task allowed if no direct application | CLIP label audit (`clip_label_audit.py`): train/val only, never edits labels. Direct application exists here, so present this as an extension unless the lecturer approves it as the adjacent task | Code ready |
| Document model/version, configuration, prompts, external services, limitations, costs | model names/tags in the logs and `q6_models.yaml`; weights come from the Hugging Face hub (internet ON once in Kaggle); no paid API used | Manual: verify licences |
| Lecturer's point: transformer treatment anchored to your data | `vit_facts.py` -> tokens/sequence length at your input size (576 patches, 577 with [CLS] at 384), attention cost, parameters, share of defects smaller than a patch | Ready |

## Q6.b Comparative analysis and innovation (10 marks)
| Brief bullet | Covered by | Status |
|---|---|---|
| Compare using evidence: performance, latency, cost per prediction, training-data needs, privacy, third-party dependence | `q6_comparison.md/.csv` (one table with all of them), `q6_models.yaml` (qualitative columns), optional `--gpu-usd-per-hour` for a notional cost | Run after freezing; verify the qualitative columns |
| Do not assume either side wins; conclusion follows the evidence | bootstrap 95% CIs and paired bootstrap differences (`q6_paired_bootstrap.csv`); rule: a CI containing 0 is not a win | Ready; write the conclusion from your numbers |
| Conditions under which each approach is appropriate | `docs/q6_report_template.md` section 7 | Manual |
| At least one realistic opportunity beyond the ANN | label audit (implemented: evidence in `label_audit*.csv/png`), plus discussion of generated inspection notes, retrieval of similar past defects, synthetic data | Manual write-up |

## Rules from the brief that apply here
- One split for everything, including ViT and CLIP (`load_gt` / the frozen split).
- Test isolation: ViT epoch chosen on validation, CLIP prompts/variants/thresholds chosen on validation, label audit uses train+val only; `q6_final_eval.py` needs `--confirm` and writes its own lock.
- Seeds fixed; if ViT results vary by seed, run seeds 42/43/44 and report mean and spread.
- Comparable effort: state how many ViT runs and CLIP combinations were tried (12 CLIP combinations are logged automatically; plan ~4 ViT runs).
- Experiment log: ViT and CLIP rows (families `vit`, `clip`) are written automatically; fill `what_was_learned`.
- Honest reporting; declare AI-tool use.
