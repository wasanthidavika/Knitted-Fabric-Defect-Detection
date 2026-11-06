# Question 3 – Model Development, Evaluation and Optimization

**Project:** Fabric Defect Detection using the ISL-Knit Dataset  
**Primary intended user:** Fabric quality inspector  
**Primary decision:** Identify and localise defects in fabric images  
**Primary evaluation metric:** mAP@0.5  
**Dataset split:** Train 2,365 images, validation 504 images, test 506 images  
**Random seed for the initial experiment:** 42

> **Results status:** This report is a complete draft of the Q3 methodology and analysis structure. The formal E00 training was running when this report was prepared. Numerical results, learning-curve interpretations, baseline performance, final-model results and error-analysis findings are deliberately left as clearly marked fields. They must be filled from the actual experiment outputs. No performance values have been invented.

## 3.a Development, Baseline Comparison and Error Analysis

### 3.a.1 Initial neural model and development

The initial artificial neural network (ANN) is YOLOv8s, a convolutional neural network (CNN)-based object-detection model. It was selected because the task requires both defect classification and localisation. A classification-only model would identify a defect category but would not provide the defect's position in the fabric image.

The model uses COCO-pretrained YOLOv8s weights as its starting point and is fine-tuned for the seven ISL-Knit defect classes: Loop Mark, Needle Mark, Set Up, Oil Spot, Hole, Contamination and Yarn Missing. The input image size is 640 × 640 pixels. The model contains a convolutional backbone for feature extraction, a feature-pyramid/Path Aggregation structure for combining features at different scales, and an anchor-free detection head that predicts bounding boxes and class scores.

The initial experiment, E00, uses AdamW with an initial learning rate of 0.001, batch size 2, a maximum of 30 epochs and random seed 42. The remaining augmentation and training settings use the values recorded in the run configuration. The training script uses validation during training and saves the best checkpoint according to validation performance.

**E00 result:** Validation mAP@0.5 = [INSERT FROM E00 LOG]. Best epoch = [INSERT]. Actual training time = [INSERT].

The planned development process changes one main factor at a time where practical. E01 compares SGD with momentum against the AdamW optimiser used in E00. E02 and E03 investigate lower and higher learning rates. E04 and E05 investigate batch size. E06 and E07 compare model sizes. E08–E11 investigate additional training choices defined in the experiment plan. The final configuration must be selected using validation results only.

| Run | Planned change from E00 | Validation mAP@0.5 | Best epoch | Status / log reference |
|---|---|---:|---:|---|
| E00 | YOLOv8s, AdamW, lr 0.001, batch 2 | [ADD] | [ADD] | [ADD] |
| E01 | SGD with momentum | [ADD] | [ADD] | [ADD] |
| E02 | Lower learning rate, 0.0003 | [ADD] | [ADD] | [ADD] |
| E03 | Higher learning rate, 0.003 | [ADD] | [ADD] | [ADD] |
| E04 | Batch size 1 | [ADD] | [ADD] | [ADD] |
| E05 | Batch size 4, subject to GPU memory | [ADD] | [ADD] | [ADD] |
| E06 | Smaller YOLOv8n model | [ADD] | [ADD] | [ADD] |
| E07 | Larger YOLOv8m model, subject to GPU memory | [ADD] | [ADD] | [ADD] |
| E08–E11 | Freeze, resolution, mosaic and class-loss experiments | [ADD] | [ADD] | [ADD] |

**Figure 3.1.** Initial E00 training output and completed run summary.  
[Insert SS26 after E00 completes. If unavailable, retain this caption and add the figure later.]

**Figure 3.2.** YOLOv8s model architecture summary.  
[Insert the architecture summary or architecture diagram when available.]

Any changes made to the original plan, including reduced epochs or omitted experiments, must be stated here with the reason. Screening runs with fewer epochs must be identified as screening runs and must not be presented as equal-budget comparisons with the 30-epoch E00 run.

### 3.a.2 Classical machine-learning baseline

A classical computer-vision baseline is used to determine whether a traditional texture-based method can detect the same fabric defects. The baseline combines handcrafted texture descriptors with an RBF Support Vector Machine (SVM) in a sliding-window detection pipeline.

The feature set contains:
- Gabor filters using three frequencies and four orientations to capture directional texture patterns.
- Gray-Level Co-occurrence Matrix (GLCM) features using four properties, two distances and four angles.
- Local Binary Patterns (LBP), using the uniform pattern with P = 8 and R = 1.

The detector generates candidate windows from anchors derived by k-means clustering of training-set bounding boxes. It uses a two-stage cascade, ridge regression for bounding-box refinement and per-class non-maximum suppression (NMS). The planned configuration uses K = 300 candidate windows per image and a sliding-window stride equal to half the shorter image side, with a minimum stride of 8 pixels.

The planned SVM search space contains C ∈ {1, 10, 100} and gamma ∈ {0.001, 0.01, 0.1}, giving nine SVM configurations. Three per-class NMS IoU settings, {0.3, 0.4, 0.5}, give a total of 12 planned baseline configurations. These settings are selected using validation mAP@0.5. The ANN plan also contains 12 planned screening configurations. The report must state the number actually completed, not only the planned number.

The following settings are fixed rather than tuned: K, first-stage NMS IoU of 0.5, ridge alpha of 1.0, five anchors, a 1:1 background-to-positive window ratio and a maximum of 3,000 positive windows per class.

**Implementation deviation:** The baseline is designed to use the same prepared 640-pixel images as YOLO, rather than the 1024-pixel images considered in the earlier dataset proposal. This makes the model inputs more directly comparable and avoids repeatedly reading the original large images. The trade-off is that very small defects may lose visual detail.

| Baseline result | Value |
|---|---:|
| Best validation configuration | [ADD] |
| Number of configurations planned | 12 |
| Number of configurations completed | [ADD] |
| Validation mAP@0.5 | [ADD] |
| Validation mAP@0.5 without box regression | [ADD] |
| Cascade recall upper bound | [ADD] |
| Training time | [ADD] |
| Inference latency per image | [ADD] |

**Figure 3.3.** Classical baseline feature extraction and SVM grid-search results.  
[Insert caption now. Add the actual grid-results or baseline-summary screenshot after the baseline finishes.]

### 3.a.3 Learning curves and model limitations

Learning curves are used to assess whether the model is learning effectively and whether additional training is useful. The training and validation losses, validation mAP@0.5 and validation mAP@0.5:0.95 should be reviewed together.

A rising validation mAP near the final epoch may indicate that the model is still improving and that the training budget is limited. A falling training loss accompanied by worsening validation loss may indicate overfitting. If both training and validation performance remain poor, the model may be underfitting or limited by image resolution, label quality or the available data.

YOLO training loss is calculated on augmented training images. Mosaic augmentation changes the training image distribution, so training loss should not be compared mechanically with validation loss. The learning-curve plot should be interpreted with the epoch at which mosaic augmentation is disabled taken into account.

The diagnosis must be based on the generated learning-curve plots and `learning_curve_diagnosis.csv`.

| Diagnostic question | Evidence to insert | Interpretation |
|---|---|---|
| Was validation mAP still increasing in the final five epochs? | [ADD final-five-epoch slope] | [ADD] |
| Did validation loss rise while training loss fell? | [ADD minimum and final validation loss] | [ADD] |
| Did a larger model improve performance? | [ADD E06, E00 and E07 results] | [ADD] |
| Did freezing layers reduce or improve learning? | [ADD E08 result] | [ADD] |
| Was image resolution a limitation? | [ADD E09 result] | [ADD] |

**Figure 3.4.** E00 training and validation learning curves.  
[Insert `learning_curve_E00.png` when generated.]

**Figure 3.5.** Final selected model learning curves.  
[Insert the final model learning-curve figure when generated.]

The final diagnosis is [UNDERFITTING / OVERFITTING / EPOCH-LIMITED / RESOLUTION-LIMITED / OTHER]. This conclusion is based on [INSERT specific curve values and experiment IDs]. No diagnosis should be selected until the real curves are available.

### 3.a.4 Comparison with the reference benchmark and classical baseline

A trivial reference detector is included to show the difficulty of the localisation task. It predicts one whole-image bounding box and assigns the most frequent defect class. Since the ground-truth defects occupy only parts of the image, a whole-image box is expected to achieve very low IoU-based detection performance.

All models should be evaluated with the same metric implementation and the same fixed data split. The primary metric is mAP@0.5. mAP@0.5:0.95 is included as a stricter secondary metric. The validation set is used for model selection and threshold selection. The held-out test set is reserved for the final evaluation.

| Model | mAP@0.5 | mAP@0.5:0.95 | Training time (min) | Inference latency (ms/image) |
|---|---:|---:|---:|---:|
| Trivial whole-image reference | [ADD] | [ADD] | 0 | approximately 0 |
| Classical Gabor + GLCM + LBP + SVM | [ADD] | [ADD] | [ADD] | [ADD] |
| Initial ANN, E00 | [ADD] | [ADD] | [ADD] | [ADD] |
| Final ANN, mean ± standard deviation over seeds 42, 43 and 44 | [ADD] | [ADD] | [ADD] | [ADD] |

The final ANN row may only report a mean and standard deviation if all three final seed runs have actually been completed. Otherwise, report the available result and clearly state that the three-seed evaluation remains incomplete.

The report must state the hardware used, including the CPU and GPU. The current development machine uses an NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM. Confirm the exact hardware recorded for each formal run before using this statement in the final comparison. ANN training time should include the training run and its per-epoch validation. Baseline training time should state whether feature extraction, stage-one training and SVM fitting are included, and whether grid-search time is excluded. Inference latency should be measured at batch size one and should state whether image decoding and preprocessing are included.

**Figure 3.6.** Final comparison of the reference detector, classical baseline, initial ANN and final ANN.  
[Insert the generated comparison table or chart after final evaluation.]

### 3.a.5 Error analysis

Error analysis is performed on the selected final model to identify the types of fabric defects that are difficult to detect. The analysis should use the saved predictions and report results by defect class, defect size, fabric colour, crowding and image quality where the available data supports those comparisons.

The dataset contains seven defect classes. The audit also identified small defects as a potential challenge: at 640-pixel input size, the median short-side defect dimension was approximately 14.4 pixels, and 13.4% of defects were below 8 pixels. This may make small defects difficult to localise accurately. The dataset audit also identified a possible colour shortcut: grey/dyed fabric distribution was associated with class labels, and colour-only logistic models achieved ROC-AUC above 0.65 for some classes, including Set Up (0.772) and Needle Mark (0.748). These are dataset audit findings, not evidence that the final YOLO model necessarily uses colour as a shortcut.

The final error analysis should report:
- Which classes have the lowest recall and which classes are confused with one another.
- Whether recall decreases for small bounding boxes.
- Whether performance differs between grey and dyed fabric.
- Whether crowded images or overlapping defects produce more missed detections.
- Common false positives and false negatives.
- Whether difficult examples appear to involve low image quality, unusual fabric appearance, ambiguous labels or annotation problems.

| Error-analysis area | Result to insert |
|---|---|
| Lowest-recall defect classes | [ADD from per-class results] |
| Most frequent class confusions | [ADD from confusion matrix] |
| Recall by defect size | [ADD] |
| Grey versus dyed fabric performance | [ADD] |
| Performance by crowding | [ADD] |
| Main false-positive patterns | [ADD] |
| Main false-negative patterns | [ADD] |

**Figure 3.7.** Final-model confusion matrix and per-class performance.  
[Insert after final error analysis.]

**Figure 3.8.** Examples of correct detections, false positives and missed defects.  
[Insert representative validation or final-test examples, clearly identifying which split each image comes from.]

Possible future improvements include collecting more examples of under-represented defects, improving annotations for ambiguous cases, increasing input resolution if computational resources permit, and evaluating the model on fabric rolls or dye lots that were not represented during training. Roll or dye-lot identifiers were not available for the current split, so complete separation by physical roll or dye lot cannot be confirmed.

## 3.b In-depth Study of Optimization Techniques

### 3.b.1 Optimiser comparison

The initial ANN uses AdamW. E01 is designed to compare it with SGD using momentum 0.937 and an initial learning rate of 0.01.

AdamW adapts the update size for each parameter using estimates of the first and second moments of gradients. This can support faster early optimisation and is useful when fine-tuning pretrained networks. Its performance can be sensitive to the learning rate, particularly when the learning rate is too high for pretrained weights. AdamW also applies weight decay separately from the adaptive gradient update.

SGD with momentum uses a shared learning-rate scale and accumulates information from previous gradients. Momentum can smooth updates and maintain progress in directions where gradients are consistent. SGD may require more careful learning-rate selection and may converge more slowly at the beginning, but it can generalise well when appropriately tuned.

The comparison must be based on the actual E00 and E01 validation results and curves.

| Run | Optimiser | Initial learning rate | Validation mAP@0.5 | Best epoch | Observed curve behaviour |
|---|---|---:|---:|---:|---|
| E00 | AdamW | 0.001 | [ADD] | [ADD] | [ADD] |
| E01 | SGD with momentum 0.937 | 0.01 | [ADD] | [ADD] | [ADD] |

In this experiment, [INSERT OPTIMISER] achieved the higher validation mAP@0.5 by [ADD DIFFERENCE]. The learning curves show [ADD OBSERVATION]. This suggests that [EXPLAIN USING THE OBSERVED CURVES]. The conclusion is limited by the use of one seed in each screening run. Small differences may be due to random variation rather than the optimiser itself.

### 3.b.2 Hyperparameter investigation

Three main hyperparameter groups are planned: learning rate, batch size and model size. Each group is compared using validation performance. Other settings should remain unchanged where possible.

| Hyperparameter | Runs | Planned values | Result and conclusion |
|---|---|---|---|
| Learning rate | E02, E00, E03 | 0.0003, 0.001, 0.003 | [ADD actual results and best setting] |
| Batch size | E04, E00, E05 | 1, 2, 4 | [ADD actual results and memory/training observations] |
| Model size | E06, E00, E07 | YOLOv8n, YOLOv8s, YOLOv8m | [ADD actual results, speed and capacity observations] |

A lower learning rate may provide more cautious updates when fine-tuning pretrained weights, but it may make learning slower. A higher learning rate may improve early progress but can cause unstable updates or reduce useful pretrained features. Batch size affects the number of samples used for each gradient update and the memory required. Model size affects representational capacity, computation and inference speed.

The conclusions must be based on the measured validation results, training time and resource use. A difference from one screening run should be treated cautiously because each screening configuration uses one seed. The final three-seed runs, if completed, provide an estimate of variation due to random initialisation and training randomness.

**Figure 3.9.** Validation mAP comparison across learning-rate experiments E02, E00 and E03.  
[Insert after the runs finish.]

**Figure 3.10.** Validation mAP comparison across batch-size experiments E04, E00 and E05.  
[Insert after the runs finish.]

**Figure 3.11.** Validation mAP and model-size comparison across E06, E00 and E07.  
[Insert after the runs finish.]

### 3.b.3 Approaches that did not work and diagnosis

The experiment plan includes several alternatives that can reveal limitations in the training approach. An experiment should only be described as unsuccessful if its actual result supports that conclusion. A run that performs well must not be presented as a failure.

At least two substantive unsuccessful approaches are required by the coursework. Suitable candidates include freezing part of the backbone, increasing input resolution to 960 pixels, disabling mosaic augmentation, or increasing the classification loss gain. The final selection must come from experiments that were actually run.

For each unsuccessful approach, the analysis should explain the original hypothesis, the observed result, the reason it was less suitable for this dataset and what was learned.

| Experiment | Hypothesis before the run | Actual outcome | Evidence-based diagnosis |
|---|---|---|---|
| [E08 or other completed run] | [COPY hypothesis from experiment plan] | [ADD actual metric and curve evidence] | [ADD diagnosis] |
| [E09/E10/E11 or other completed run] | [COPY hypothesis from experiment plan] | [ADD actual metric and curve evidence] | [ADD diagnosis] |

For each important rejected alternative, state what characteristics would have made it appropriate and what property of this problem made it less suitable. For example, freezing the backbone may be appropriate when pretrained visual features already match the target domain and the dataset is very small. It may be less suitable if fabric textures and defect patterns differ substantially from the pretraining data and the model needs to adapt its feature representations. This is a hypothesis to assess against the actual E08 result, not a claim that E08 has already failed.

Increasing image resolution may be appropriate when small defects are lost at the input scale and sufficient GPU memory is available. It may be less suitable on a 4 GB GPU if the larger input causes memory errors, substantially slower training or a smaller batch size that makes optimisation less stable. This explanation should only be used if supported by the E09 run.

Disabling mosaic augmentation may be appropriate when synthetic image combinations distort the target objects or when the model is being fine-tuned near convergence. It may be less suitable if the training data benefits from the additional variation created by mosaic. The E10 curves and validation result must determine the conclusion.

Increasing the classification loss gain may be appropriate when class prediction is the main weakness. It may be less suitable if localisation quality is the main problem or if stronger classification emphasis reduces the balance between classification and box regression. The E11 result must be used to decide whether this occurred.

**Figure 3.12.** Evidence from at least two completed unsuccessful experiments.  
[Insert the relevant learning curves, run summaries or failure screenshots after the experiments are completed.]

### 3.b.4 Experiment logging and reproducibility

All substantive training and baseline experiments are recorded in `logs/experiments.csv`. The log should include the experiment ID, attempt number, configuration, seed, hardware, execution status, scientific outcome, validation metrics and what was learned. Every experiment discussed in this report must have a corresponding log entry. Every completed log entry should contain a meaningful `what_was_learned` explanation.

The dataset is divided into fixed training, validation and test partitions. The current split contains 2,365 training images, 504 validation images and 506 test images. The split verification found 3,328 image groups and no groups spanning multiple partitions. Duplicate image IDs/stems were not found across partitions. Roll and dye-lot identifiers were unavailable, so grouping by physical roll or dye lot could not be verified.

The validation set is used for hyperparameter selection, model selection and detection-threshold selection. The test set is reserved for one final evaluation after the models and thresholds are frozen. The final evaluation script creates a lock file to prevent accidental repeat evaluation. Test results must not be used to modify the model, prompts or thresholds.

The initial experiment uses seed 42. The final model is intended to be evaluated with seeds 42, 43 and 44 if time and resources permit. If the three runs are completed, report the mean and standard deviation. If not, state the limitation clearly.

The planned ANN screening budget contains 12 configurations. The classical baseline has 12 planned configurations, consisting of nine SVM combinations and three NMS settings. The report must distinguish planned configurations from configurations actually completed.

**Figure 3.13.** Experiment log showing completed and unsuccessful attempts.  
[Insert a screenshot or table from `logs/experiments.csv` after the experiments are complete.]

### 3.b.5 Summary

The project investigates a CNN-based object detector and a handcrafted-feature SVM detector for fabric defect localisation. YOLOv8s is used as the initial ANN because it predicts both defect classes and bounding-box locations. The classical baseline provides a traditional computer-vision comparison. The optimisation study examines optimiser choice, learning rate, batch size and model size, while additional experiments investigate training and resolution choices.

The final conclusion must be based on the completed validation and held-out test results. It should identify which model performed best, the practical trade-offs in training and inference time, the main defect types that remain difficult, and the limitations caused by small defects, possible colour-related dataset bias, unavailable roll identifiers and the size of the available dataset.

## Appendix for Question 3

Include the following:
1. `logs/experiments.csv` with a `what_was_learned` entry for every experiment.
2. The final experiment configuration files and the selected model configuration.
3. The generated learning curves and learning-curve diagnosis table.
4. The classical baseline summary and grid-search results.
5. The final comparison table and error-analysis outputs.
6. A brief declaration of AI-tool use, following the institution's coursework rules.

## Screenshot and figure placement checklist

The captions above are deliberately included now. Add the actual screenshots or figures only after the relevant work has been completed.

| Figure | Evidence to capture | When |
|---|---|---|
| 3.1 | E00 completed training output and saved best checkpoint | After E00 |
| 3.2 | YOLOv8s architecture summary | Already available or generate once |
| 3.3 | Classical baseline grid results and summary | After baseline |
| 3.4 | E00 learning curve | After curve generation |
| 3.5 | Final ANN learning curve | After final model |
| 3.6 | Four-model comparison table | After final evaluation |
| 3.7 | Confusion matrix and per-class results | After final error analysis |
| 3.8 | Correct, false-positive and false-negative examples | After error analysis |
| 3.9 | Learning-rate comparison | After E02/E03 |
| 3.10 | Batch-size comparison | After E04/E05 |
| 3.11 | Model-size comparison | After E06/E07 |
| 3.12 | At least two genuinely unsuccessful substantive experiments | After those runs |
| 3.13 | Completed experiment log | At report finalisation |

**Before submission:** Replace every `[ADD]` and bracketed instruction with values from the actual outputs, remove any experiment that was not run from claims of completed work, and make sure every result has a matching experiment-log ID. Do not use the test set for further tuning.
