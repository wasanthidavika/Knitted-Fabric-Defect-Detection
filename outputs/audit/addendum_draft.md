# Addendum: Dataset Audit and Firm Decisions
**Project:** AI-Based Knitted Fabric Defect Detection and Explainable Quality Inspection
**Dataset:** ISL-Knit (Kaggle), audited with a script included in my submission (audit_isl_knit.py, seed 42).

## 1. Verified dataset facts
Licence: [FILL: copy exactly from the Kaggle page / paper]. Source paper: [FILL: citation].
3375 images in total; labels are YOLO bounding boxes, 20679 annotated instances, 7 defect classes (class_0, class_1, class_2, class_3, class_4, class_5, class_6).
Most common image sizes: {'4000x3000': 978, '4608x3456': 566, '4608x2592': 494, '3000x4000': 227, '4160x3120': 176}.

| split | images | with_label_file | no_defect_images | boxes |
|---|---|---|---|---|
| train | 2018 | 2018 | 205 | 12195 |
| val | 326 | 326 | 46 | 2264 |
| test | 1031 | 1031 | 106 | 6220 |

Images per class (an image counts once per class it contains):

| cls_name | train | val | test | total |
|---|---|---|---|---|
| class_0 | 175 | 25 | 85 | 285 |
| class_1 | 217 | 48 | 114 | 379 |
| class_2 | 319 | 32 | 153 | 504 |
| class_3 | 540 | 93 | 257 | 890 |
| class_4 | 501 | 84 | 287 | 872 |
| class_5 | 389 | 63 | 190 | 642 |
| class_6 | 193 | 24 | 90 | 307 |

Label-quality issues found: none. Unreadable images: 0. Labels without images: {'test': 0, 'train': 0, 'val': 0}.
Images with no defect annotation: 10.6% of all images. Images with more than one defect class: 662 (21.9% of defective images).

## 2. Leakage and split strategy
30 near-duplicate image pairs (dHash Hamming distance <= 12/256) cross split boundaries; the share of images with a near-duplicate in another split is train: 1.0%, val: 1.5%, test: 1.3%. Exact (MD5) duplicate groups: 1 (1 spanning splits). 2 filename-derived groups (augmentation/export suffixes removed) appear in more than one split. 
No metadata file with fabric-roll or dyeing-lot identifiers was found, so roll-level grouping cannot be enforced directly.

**Decision:** Because related images cross the supplied split boundaries, I will NOT use the supplied split. I will build one fixed, group-aware split (groups = connected components of near-duplicate, exact-duplicate and same-filename-stem links; 3328 groups), stratified by rarest defect class, using StratifiedGroupKFold (seed 42, about 70/15/15). It is frozen before any model tuning.

## 3. Grey vs dyed fabric
Grey/dyed was inferred ({'saturation_otsu(HEURISTIC)': 3375}). Only 0% of paths carry a colour keyword, so the rest uses a mean-saturation threshold (0.2832); verified by eye on colour_check.png [EDIT: confirm]. Counts: {'grey': 2921, 'dyed': 454}. Class-by-colour image counts:

| cls_name | dyed | grey | dyed_share |
|---|---|---|---|
| class_0 | 33 | 252 | 0.116 |
| class_1 | 40 | 339 | 0.106 |
| class_2 | 104 | 400 | 0.206 |
| class_3 | 38 | 852 | 0.043 |
| class_4 | 163 | 709 | 0.187 |
| class_5 | 56 | 586 | 0.087 |
| class_6 | 18 | 289 | 0.059 |

chi-square test of class vs colour p = 1.18e-28. [EDIT: one sentence interpreting this: is any class almost only grey or only dyed? If yes, state it is a shortcut risk.]
**Decision:** I will (a) stratify the split by class and report colour balance per split, (b) train a colour-only logistic regression on mean RGB/HSV as a shortcut check; if it scores well above the trivial benchmark, colour is a confound, and (c) report error analysis separately for grey and dyed fabric.

## 4. Three findings that constrain modelling
1. **Class imbalance:** the most frequent class (class_3, 890 images) is 3.12x the rarest (class_0, 285). Accuracy is therefore unsuitable; I will use class-weighted loss/sampling and macro-averaged metrics.
2. **Defect size vs input resolution:** median defect short side is 6.3 px at 224x224 and 14.3 px at 512x512; the share of defects with a short side under 8 px is 65.1% at 224, 29.5% at 384 and 15.5% at 512. The audit suggests an input of at least 640 px. [EDIT: confirm after viewing sample images]
3. **Leakage / confounds:** cross-split near-duplicates exist, so a group-aware split is mandatory; class-vs-colour association is documented in section 3; 10.6% of images contain no defect, which defines how the "no defect" outcome is handled.

## 5. Firm decisions
- **Task:** image-level multi-label defect classification derived from the YOLO boxes (class present / not present; images without boxes are "no defect"). Localisation is provided by Grad-CAM, not by a detector. [EDIT: change if you decide otherwise, and say why]
- **User and decision:** a fabric quality inspector decides whether a fabric image is passed or sent for manual re-inspection, and which defect type to look for.
- **Error costs:** a missed defect (false negative) lets faulty fabric proceed to dyeing/cutting; a false alarm costs only an extra manual check. False negatives are therefore costlier. [EDIT: confirm with your own reasoning]
- **Primary metric:** macro-F1 on the held-out test set (per-class recall also reported); the decision threshold is chosen on validation data only.
- **Classical baseline:** Gabor filter bank + GLCM + LBP features, concatenated and standardised, into an RBF-SVM tuned by grid search over C and gamma [FILL: grid, number of configurations]; plus gradient boosting on the same features as a second classical model. The ANN gets a tuning budget of the same order of magnitude [FILL: number of configurations for each].
- **ANN:** pretrained CNN fine-tuned (ResNet18 first candidate) at input resolution [FILL: 224/384/512 from finding 2] with class-weighted cross-entropy (or per-class sigmoid/BCE, because images can contain several defects).
- **Vision Transformer (transformer treatment):** a ViT-B/16 receives the same images. At 224x224 with 16x16 patches the sequence is (224/16)^2 = 196 patch tokens (+1 [CLS] = 197); at 384x384 it would be 576 tokens (+1 = 577). Given finding 2, I will also evaluate the higher resolution because 224 px may erase small defects. Attention cost grows quadratically with token count, which is the main practical limit.
- **Question 6:** a pretrained vision-language model is applied to the same held-out test images with the same macro-F1; prompts are selected on validation data only; model/version, prompts, cost and latency are documented. [EDIT: name the model]
- **Explainability:** Grad-CAM on the CNN, validated by occluding/perturbing highlighted regions and checking the prediction falls; incorrect predictions analysed.

## 6. Question 2.b prediction (made before any results)
[EDIT: write your own: which family do you expect to win on macro-F1, and why? Use the audit: dataset size, texture-dominated images, defect size, imbalance.]

## 7. How my work differs from the dataset paper
[EDIT: after reading the paper, state its task, baselines and metrics, and how your task, split, baseline and evaluation differ.]
