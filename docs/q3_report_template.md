# Q3 - Model development, evaluation and optimisation (report skeleton, ~6 pages)
Quote only numbers that exist in logs/experiments.csv or outputs/q3/. Every result in the text must trace to a log id.

## 3.a Development, baseline comparison and error analysis

### 1. Initial neural model and how it was improved
- Initial ANN = run E00: YOLOv8s, COCO-pretrained, 640 px, AdamW lr0=0.001, batch 8, 30 epochs, seed 42, defaults for the rest [state the augmentation defaults]. Val mAP@0.5 = [E00].
- Improvement path: [list log ids in order with the change and the val effect]. Final configuration: [ ]; trained with seeds 42/43/44 for [N] epochs.
- Deviations from the plan: [anything you changed after seeing results, and why].

### 2. Classical baseline
- Design: Gabor (3 frequencies x 4 orientations) + GLCM (4 properties x 2 distances x 4 angles) + LBP (uniform, P=8 R=1) -> RBF-SVM, sliding-window detector, two-stage cascade, ridge box regression, per-class NMS.
- Windows: [anchors printed by baseline_train.py] from k-means on TRAINING boxes; stride half the shorter side (min 8 px); K = 300 candidates per image.
- Search space: C in {1, 10, 100} x gamma in {0.001, 0.01, 0.1} = 9 SVM configurations, + per-class NMS IoU in {0.3, 0.4, 0.5} = 3 post-processing settings => 12 configurations, selected on validation mAP@0.5. ANN: 12 planned runs. Same order of magnitude.
- Fixed, not tuned (state them): K, stage-1 NMS IoU 0.5, ridge alpha 1.0, 5 anchors, 1:1 background:positive windows, at most 3000 positive windows per class.
- Results: best config [id], val mAP@0.5 [ ], without box regression [ ], cascade recall upper bound [ ] (cap on what stage 2 can recover).
- DEVIATION from the addendum (say it explicitly): the baseline runs on the SAME 640 px images as YOLO (not 1024 px). Reason: identical input for both models and no re-reading of the 15 GB originals; consequence: smaller defects.

### 3. Learning curves (evidence-based interpretation)
Insert `learning_curve_E00.png` and the final run's. Use `learning_curve_diagnosis.csv`. Answer with numbers:
- Is val mAP still rising at the last epoch (budget-limited)? slope of last 5 epochs = [ ].
- Does val loss rise while train loss falls (overfitting)? min val loss at epoch [ ] vs final [ ].
- Remember train loss is measured on mosaic-augmented images, so train loss > val loss is expected; the dotted line marks where mosaic switches off.
- Capacity: compare E06 (n) / E00 (s) / E07 (m) and E08 (frozen backbone): if a bigger model clearly improves, the small model was capacity-limited; if not, data or resolution limits it (E09).
- Conclusion: [underfitting / overfitting / epoch-limited / resolution-limited] because [evidence].

### 4. Comparison table (from final_comparison.md)
| Model | mAP@0.5 (primary) | mAP@0.5:0.95 | Training time (min) | Inference latency (ms/image) |
|---|---|---|---|---|
| Trivial: whole-image box of the most frequent class | | | 0 | ~0 |
| Classical baseline (Gabor+GLCM+LBP -> SVM) | | | | |
| Initial ANN (E00) | | | | |
| Final ANN (mean +/- std over 3 seeds) | | | | |
State: hardware (CPU/GPU name), what training time includes (baseline: stage-1 + train features + best SVM fit, grid search excluded; ANN: wall-clock incl. validation each epoch), latency scope (batch 1, decoding excluded). The trivial benchmark scores ~0 because small boxes cannot reach IoU 0.5 with a whole-image box: say so, it shows detection is not trivial here.
Metric implementation: all rows scored by src/metrics.py; for YOLO state the difference to Ultralytics' own value on validation (evaluate_detections.py --compare-log-id).

### 5. Error analysis of the final model (from outputs/q3/error_analysis/)
Use the numbers and figures; answer: classes missed/confused (confusion matrix), recall by defect size (links to Finding 1), grey vs dyed (links to the colour-shortcut result), crowding, false positives, worst cases (label noise? duplicates? unusual images?). Say what could be done next WITHOUT claiming it was done.

## 3.b Optimisation study

### 6. Optimisers
| Run | Optimiser | lr0 | val mAP@0.5 | best epoch | curve behaviour |
|---|---|---|---|---|---|
| E00 | AdamW | 0.001 | | | |
| E01 | SGD+momentum 0.937 | 0.01 | | | |
Relate to mechanics: SGD with momentum accumulates gradients along consistent directions and needs a larger lr, per-parameter step size is global; Adam/AdamW scale each parameter's step by its gradient history (faster early progress, sensitive to lr on pretrained weights), AdamW decouples weight decay from that scaling. Which behaviour do YOUR curves show? [evidence].

### 7. Hyperparameters (at least three)
| Hyperparameter | Runs | Results | Conclusion supported by |
|---|---|---|---|
| Learning rate | E02 (0.0003), E00 (0.001), E03 (0.003) | | |
| Batch size | E04 (4), E00 (8), E05 (16) | | |
| Depth / width | E06 (n), E00 (s), E07 (m) | | |
One seed per screening run: say that differences smaller than the seed spread (measure it with the final 3 seeds) are not conclusions.

### 8. Approaches that did not work (at least two, diagnosed)
For each (E08 frozen backbone, E09 960 px, E10 no mosaic, E11 class-loss gain, or any other run that really failed):
Hypothesis (written BEFORE the run, copied from q3_plan.csv) -> what happened (numbers) -> diagnosis (evidence from curves, flags, error analysis) -> what it tells us.
A run that simply worked as hoped is not a failure: report honestly which runs failed.

### 9. Experiment log
Attach logs/experiments.csv as an appendix. Check: every run in this chapter has a row; every row has `what_was_learned`; numbers in the text equal the log.
