# Q6 run order (Kaggle, free) and file placement

## File placement
| File | Goes to |
|---|---|
| `src/image_level.py, vit_facts.py, vit_train.py, clip_zeroshot.py, clip_label_audit.py, q6_final_eval.py` | `src/` |
| `configs/q6_prompts.yaml, q6_models.yaml` | `configs/` |
| `tests/test_image_level.py, test_clip_pipeline.py, run_q6_smoke.sh` | `tests/` |
| `docs/*.md` (incl. the updated `final_checklist.md`, which REPLACES the old one) | `docs/` |
Nothing from Q1-Q5 is modified.

## Before you start
- `pip install timm transformers` if missing (Kaggle usually has them). Turn Internet ON in the notebook once so the weights can download (free); then CLIP/ViT run locally.
- Check the ViT weight tag: `python -c "import timm; print(timm.list_models('vit_base_patch16_384*', pretrained=True))"` and put the exact name in `--model` if it differs.
- Tests anywhere: `python tests/test_image_level.py && python tests/test_clip_pipeline.py`. Full pipeline test with a fake scorer: `bash tests/run_q6_smoke.sh` (after `run_q3_smoke.sh`).
- Look at `outputs/q1/figures/examples_per_class.png` and EDIT `configs/q6_prompts.yaml` so every description matches what the defect really looks like. Prompt wording is decided on validation images only.

## Steps (GPU session for ViT/CLIP; times are guesses, NOT measured)
```
python src/vit_facts.py                                   # tokens, sequence length, params, defects smaller than a patch
python src/clip_zeroshot.py --stage select                # val only: 12 combinations, all logged; writes clip_chosen.json
python src/clip_zeroshot.py --stage latency
python src/vit_train.py --name V00                        # ~tens of minutes per run on a T4 (unmeasured): time one epoch first
python src/vit_train.py --name V01 --model vit_base_patch16_224.augreg_in21k_ft_in1k
python src/vit_train.py --name V02 --freeze-backbone
python src/vit_train.py --name V03 --lr 1e-5
python src/clip_label_audit.py                            # train+val crops only; then review ~50 flagged + ~50 random crops by eye
```
If the chosen ViT config varies a lot between runs, repeat it with `--seed 43` and `--seed 44` (new `--name` each time) and report mean and spread.

## The test evaluation (everything frozen; do it in ONE session)
```
python src/final_eval.py --confirm ...                    # Q3 models (writes outputs/q3/preds/*_test.csv)
python src/q6_final_eval.py --confirm \
   --yolo-val outputs/q3/preds/F42_val.csv --yolo-test outputs/q3/preds/final_ANN_F42_test.csv --yolo-label final_ANN_F42 \
   --svm-val outputs/q3/preds/SVM_BEST_val.csv --svm-test outputs/q3/preds/classical_SVM_baseline_test.csv \
   --vit-weights outputs/q6/vit/V00/best.pt --vit-name V00 --clip
```
It has its own lock file. Do not re-run after seeing the numbers.

## Known limits (read before trusting)
- `vit_train.py` and `HFClipScorer` were NOT run (no torch/timm/transformers in the authoring sandbox). Everything around them is tested: image-level labels/scores/thresholds/bootstrap, ViT arithmetic (parameter count matches the published ViT-B/16 figure), tiling, aggregation, prompt ensembling, neighbour voting, the final-evaluation locks. Expect small fixes on the first real run and send me the traceback.
- Tests used a FAKE CLIP scorer: they prove plumbing, not CLIP quality or ViT accuracy.
- Image-level scoring is a different task from detection: a ViT or CLIP that "wins" at image level still cannot show the inspector where the defect is.
- The qualitative columns in `q6_models.yaml` (licences, sizes, dependence) are my best knowledge: verify them.
