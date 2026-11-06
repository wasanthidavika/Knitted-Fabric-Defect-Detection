# Q2 - Solution design (report skeleton, ~5 pages)
Facts about YOLOv8 below are from memory of the Ultralytics defaults: VERIFY each against `outputs/q2/param_summary.json` and the Ultralytics docs before quoting.

## 1. Architecture (Q2.a)
Insert `architecture.png`, and `q2_architecture_table.md` (layers, output shapes, parameters). State total parameters (nc=7) from `param_summary.json`, GFLOPs, strides.
Explain in your own words: Conv block (conv + normalisation + activation), C2f (cheap multi-path feature reuse), SPPF (multi-scale context), PAN-FPN neck (why stride-8 features matter for tiny defects), decoupled head.

## 2. Assumptions about the data
Local texture patterns (convolution locality), translation equivariance (defects can appear anywhere: see position heatmap), multi-scale objects (box-size figure), defects visible at the chosen resolution (Finding 1). Where the data violates an assumption, say so (e.g. [very small boxes, global lighting variation]).

## 3. Design decisions (justify each separately)
| Decision | Choice | Justification | Evidence / where tested |
|---|---|---|---|
| Depth and width | YOLOv8s (scale s) | [capacity vs 2,360 training images] | [Q3.b: n / s / m runs] |
| Input resolution | 640 px | Finding 1 | [Q3.b: 960 px run] |
| Normalisation | BatchNorm in each Conv block [verify count] | [stabilises training; depends on batch size] | [Q3.b: batch-size runs] |
| Activation | SiLU [verify `activation_types_found`] | [smooth, non-zero gradient for negatives] | - |
| Regularisation | weight decay + augmentation (mosaic, HSV jitter, flips) [verify defaults]; no dropout [verify `module_counts`] | [small dataset; HSV jitter may reduce colour reliance, see colour_shortcut] | [Q3.b: augmentation runs] |
| Output layer | 3 scales, decoupled anchor-free heads; per location 4 x reg_max box logits + nc class logits [verify `outputs_per_location`] | [multi-scale; separate tasks] | param_summary.json |
| Classification loss | BCE with logits (independent per class) [verify] | [images carry several defect classes] | - |
| Box loss | CIoU + distribution focal loss [verify] | [localisation quality] | - |
| Decision threshold | chosen on validation to reach target recall | false negatives are costlier (Q1) | [Q3.a] |
| Imbalance handling | tested as experiments (class-loss gain, oversampling) | Finding 2 | [Q3.b] |
Be honest that most of the architecture is the Ultralytics default and say which choices are yours.

## 4. Alternatives (Q2.b)
| Alternative | Would be appropriate when | Why less suitable here | Evidence |
|---|---|---|---|
| Image-level CNN classifier | defects large, presence only | tiny defects; no localisation for the inspector | resolution_vs_visibility |
| Faster R-CNN | accuracy over speed, small objects | heavier, slower to tune and explain within budget | [cite] |
| DETR-style transformer detector | very large datasets | data-hungry, 2,360 training images | [cite] |
| Anomaly detection (autoencoder / PatchCore) | few or no labelled defects | all 7 classes are labelled | - |
| Tiling + classifier | very large images, tiny defects | stitching complexity; fallback only | - |
| Texture-feature SVM | small data, interpretable features | used as the baseline | Q3 |
Why an ANN at all: [image modality, spatial structure, 20k+ labelled boxes, need for localisation].

## 5. Prediction made before any results
[Paste the addendum's section 7 unchanged, with its date. Do not edit after seeing results.]
