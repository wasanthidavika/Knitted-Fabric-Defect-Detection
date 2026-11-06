# Q1 Findings That Constrain Modelling Decisions

## Finding 1: Small Defect Regions Require Higher Input Resolution

The median defect bounding-box short-side length was 5.04 pixels at an image long-side resolution of 224 pixels and 14.4 pixels at 640 pixels.

The proportion of defect bounding boxes with a short side below 8 pixels was:

| Resolution | Defects below 8 pixels |
|---|---:|
| 224 | 75.4% |
| 384 | 44.3% |
| 512 | 24.9% |
| 640 | 13.4% |

**Modelling consequence:**

A low input resolution could make small fabric defects difficult to detect. Therefore, 640 pixels is selected as the initial detector input resolution. Higher resolution can be investigated as an additional experiment if computational resources permit.

The primary detection metric will be mAP@0.5, with mAP@0.5:0.95 as a stricter secondary metric. Per-class AP will also be reported to examine detection performance across defect categories.

## Finding 2: Significant Class Imbalance

Oil Spot represents approximately 66.7% of all annotated defect instances, whereas Needle Mark represents only 2.0%.

Defective images contain a median of one bounding box, although the mean is 7.2 boxes per image. This indicates that the distribution of defect instances across images is uneven.

**Modelling consequence:**

Overall accuracy would not adequately represent object detection performance. The evaluation will therefore include mAP@0.5, mAP@0.5:0.95 and per-class AP.

Class imbalance will also be considered when interpreting results. Any class weighting or oversampling strategies will be treated as experiments rather than assumed improvements.

## Finding 3: Duplicate Leakage and Colour-Related Confounding

The initial dataset split contained 30 cross-split near-duplicate pairs and one exact-duplicate group crossing partitions.

To address this issue, a group-aware split was generated and verified.

The final split contains:

| Partition | Images |
|---|---:|
| Training | 2,365 |
| Validation | 504 |
| Testing | 506 |
| Total | 3,375 |

The verification confirmed zero groups spanning multiple partitions and zero duplicate image IDs.

The colour-only experiment also identified potential colour-related confounding. A logistic regression model using basic colour and brightness statistics achieved ROC-AUC values above 0.65 for Loop Mark, Needle Mark, Set Up, Oil Spot and Yarn Missing.

Needle Mark achieved the highest ROC-AUC of 0.748, followed by Set Up at 0.772. These results indicate that colour and brightness alone contain information associated with several defect categories.

**Modelling consequence:**

The group-aware split will remain frozen throughout model development and evaluation. Colour-related performance differences will be investigated by evaluating results separately on grey and dyed fabrics.

The colour-only experiment will serve as a diagnostic baseline. Its results indicate a potential confounding risk, but do not prove that the final detector relies on colour.

## Dataset Summary

- Total images: 3,375
- Annotated bounding-box instances: 20,679
- Defect classes: 7
- Images without defects: 357
- Input resolution selected: 640 pixels on the long side

The five most common original image dimensions recorded in the EDA manifest were:

| Image dimensions | Number of images |
|---|---:|
| 4000 × 3000 | 984 |
| 4608 × 3456 | 571 |
| 4608 × 2592 | 494 |
| 3000 × 4000 | 221 |
| 4160 × 3120 | 176 |

These findings directly influence the selected detection resolution, evaluation metrics, split strategy and colour-related evaluation procedures.