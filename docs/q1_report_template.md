# Q1 - Dataset and Exploratory Data Analysis (report skeleton, ~4 pages)
Fill the [brackets] from `outputs/q1/`. Quote only numbers that exist in those files.

## 1. Dataset description
- Source: ISL-Knit, Kaggle `mumtaheenabinteahmed/isl-knit-defect`; paper: Das Gupta et al. (2024), Heliyon 10(17), e35931.
- Licence / permission: [exact wording from the Kaggle page; state the redistribution conditions you will respect].
- Structure: [N] images, [N] YOLO box instances, 7 classes [verified class order], image sizes [top sizes], 357 images without defects.
- Target: object detection (class + box); image-level labels derived from boxes for the ViT / CLIP comparison.
- Limitations: no roll/lot IDs, heuristic colour labels, [label noise observed in label_check / examples].

## 2. EDA (one short paragraph + figure each; say what each plot changes)
Figures in `outputs/q1/figures/`: class_distribution, boxes_per_image, box_size_by_class, resolution_vs_visibility, cooccurrence, position_heatmap, image_sizes, class_by_colour, examples_per_class.

## 3. Costs of errors
False negative: a defect is missed and faulty fabric continues to later processing (costly). False positive: an extra manual re-inspection (cheap). Consequence: [confidence threshold chosen on validation to reach a target image-level recall; justify the target].

## 4. Primary metric
mAP@0.5 (primary), with mAP@0.5:0.95, per-class AP, precision, recall and image-level recall (pass/refer) as secondary. Justify: [defects are small, annotation noise at strict IoU; inspector needs found-and-roughly-located; macro averaging because of imbalance].

## 5. Leakage checks and data quality
- Duplicates / near-duplicates: [numbers from `audit/report.json`]; method: 256-bit dHash, Hamming <= 12, MD5, filename groups.
- Group-aware frozen split: [numbers from `split_verification.json`]; no group spans two partitions.
- Label/EXIF check: [what `label_check.png` showed; chosen exif_mode].
- Colour shortcut: [`colour_shortcut.csv` result, honest interpretation].
- Handling: no-defect images kept as negatives; [outlier image sizes decision].

## 6. Three findings that constrain modelling
Copy and refine `outputs/q1/findings.md`. Each finding = evidence + the modelling decision it forces.
