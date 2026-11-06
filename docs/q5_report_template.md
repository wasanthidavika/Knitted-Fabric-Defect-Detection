# Q5 - Explainable AI (report skeleton, ~4 pages)

## 5.a Methodologies
| Method | Type | What it explains | Cost | Main limitations | Fit for a fabric inspector |
|---|---|---|---|---|---|
| EigenCAM | architecture-specific (CNN), gradient-free | where the chosen layer's features vary most (first principal component) | 1 forward pass + SVD | class-agnostic; coarse (stride 8/16/32); not tied to one detection or class; layer choice matters | quick overview, but can look "right" for the wrong reason |
| Grad-CAM / Grad-CAM++ | architecture-specific, gradient-based | gradient of a chosen score w.r.t. a feature map | 1 backward pass per target | a detector has many outputs, so a scalar target must be defined; gradients can be noisy/saturated | harder to wire to a single box |
| Occlusion sensitivity | model-agnostic, perturbation | drop in THIS detection's score when an area is hidden | many forward passes (here <= ~40) | artefacts of the fill; depends on patch size; slow | direct answer to "does the model rely on this area?" |
| LIME / KernelSHAP | model-agnostic, surrogate | local attribution over superpixels/features | hundreds of forward passes | superpixels do not fit tiny defects; unstable; slow | too slow/coarse here |
Edit this table to what YOU read and verified; keep the three or four you can explain. Justification of the selection (architecture = CNN detector; user = non-technical inspector who needs "look here" and a way to check it): [write].

## 5.b Implementation
- Occlusion: window = 1.25 x the box's longer side (so one window can hide the whole defect), explored over box +/- 1 window (cap 36 windows + the box-centred window), fill = inpainting from the surroundings (default; `--fill blur|mean` were compared: [results]), target = confidence of THE detection (best overlapping box of the same class, IoU >= 0.3).
- EigenCAM: layer [15 / 18 / 21], chosen on VALIDATION by [criterion]; letterbox padding handled; hook-based, class-agnostic.
- In the app: "Why?" button -> both maps + a plain-language sanity sentence computed live.

## Sanity checks (all on validation unless stated; numbers from `outputs/q5/deletion_summary_<tag>.json`)
1. Deletion test, N = [30] correctly detected defects: confidence drop when the top window is hidden vs random windows (same size, <= 0.2 IoU with the top window).
| Method | mean drop (top) | mean drop (random) | difference [95% CI] | top > random | Wilcoxon p |
|---|---|---|---|---|---|
| Occlusion (self-consistency) | | | | | |
| EigenCAM (independent test) | | | | | |
| Control: "just the box" | | | | | |
Interpretation rules: (a) occlusion is partly circular (built from the same kind of perturbation): it checks the pipeline, not faithfulness; (b) EigenCAM is the real test; (c) the BOX control matters: for a compact defect, hiding the box itself is the strongest single intervention. If EigenCAM does not beat the random windows, or does not beat the box control, say what the map adds beyond the detector's own box (context? nothing?). The smoke test on synthetic data showed the box control beating the occlusion window: do not assume your results look better.
2. Model-randomisation test (`--randomise`): rank correlation between the trained model's map and maps after randomising [last block / neck / whole network]: [numbers]; a map that barely changes is not explaining the model.
3. Layer / fill sensitivity: [how much do conclusions change with `--layer`, `--fill`]?

## Case studies (at least THREE incorrect predictions, each analysed)
For each figure in `outputs/q5/cases/`: image id, kind (false positive / class confusion / missed), what the model predicted and the truth, what the maps highlight, what the hiding test says, and what that suggests about the model's behaviour (e.g. reacts to texture edges, to a neighbouring defect, to label noise). Use the error-analysis worst cases to choose; do not pick only flattering examples. Add 1-2 correct detections for contrast.
| # | kind | image | predicted / truth | score | drop (top vs random) | what the explanation suggests |
|---|---|---|---|---|---|---|

## UI / UX
Screenshots; plain-language sentences used; how you made the limits of the explanation visible (caption, sanity sentence, "not available" messages); any informal feedback from a potential user.

## Limitations (state all)
Sensitivity maps are not proof of the model's reasoning or of causality; occlusion artefacts and patch-size dependence; EigenCAM is class-agnostic and coarse; maps are explanations of one image under one model; randomisation/deletion tests give evidence of sensitivity, not of correctness; explanations were validated on [N] validation detections only; results may differ for small defects (check by size bin).
