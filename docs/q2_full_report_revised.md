# Question 2 – Solution Design
**Project:** ISL-Knit Fabric Defect Detection

## 2(a) Design and Architect a Suitable ANN

### 2.1 Proposed ANN architecture

YOLOv8s was selected as the primary artificial neural network (ANN) for knitted fabric defect detection. It is a single-stage object detector that predicts defect categories and bounding-box locations in one forward pass. The model uses COCO-pretrained weights, is configured for seven defect classes, and receives RGB images resized to 640 × 640 pixels.

The architecture has three main components:

- **Backbone:** Extracts visual features using convolutional and C2f blocks. The Spatial Pyramid Pooling Fast (SPPF) module collects additional spatial context.
- **Neck:** Uses top-down and bottom-up feature fusion through a PAN-FPN structure. It combines detailed features from earlier backbone stages with higher-level semantic features.
- **Detection head:** Uses decoupled, anchor-free detection branches at three feature-map scales. It predicts bounding-box coordinates and class scores.

**Figure 2.1.** YOLOv8s architecture configured for seven ISL-Knit defect classes.  
[The architecture diagram from the original Q2 document is included below.]

### 2.2 Architecture dimensions, main layers and parameters

The detector produces three feature-map outputs with strides of 8, 16 and 32 pixels. For a 640 × 640 input, these correspond to feature maps of 80 × 80, 40 × 40 and 20 × 20. There are 8,400 prediction locations across the three scales.

| Stage / main layer group | Output dimensions | Main operation and purpose |
|---|---|---|
| Input | 640 × 640 × 3 | RGB fabric image supplied to the detector |
| Backbone: initial convolutional blocks | Progressively downsampled feature maps | Extract edges, local texture and basic visual patterns |
| Backbone: C2f blocks | 128 × 80 × 80; 256 × 40 × 40; 512 × 20 × 20 at the three main scales | Reuse and combine feature information at different depths |
| Backbone: SPPF | Deep feature representation at the coarsest scale | Adds spatial context using pooled features |
| Neck: PAN-FPN fusion | 128 × 80 × 80; 256 × 40 × 40; 512 × 20 × 20 | Combines high-resolution detail with deeper semantic information |
| Detection head | 3 scales; 8,400 locations | Predicts bounding-box distributions and seven defect-class scores |

The table summarises the main architectural stages. The complete module-by-module layer listing, including individual output shapes and parameters, should be retained from the generated `layers.csv` / model-summary output if available. The architecture diagram shows the individual indexed blocks.

At each prediction location, the raw detection output contains 71 values: 64 bounding-box distribution values (4 × reg_max, with reg_max = 16) and seven class values. The decoded prediction contains four box coordinates and seven class scores. The YOLOv8 detection head does not use a separate traditional objectness output.

| Architecture component | Parameters |
|---|---:|
| Backbone | 5,079,712 |
| Neck | 3,939,840 |
| Detection head | 2,118,757 |
| **Total** | **11,138,309** |
| Computational cost at 640 × 640 | 28.65 GFLOPs |

The model summary identifies 23 indexed main modules (indices 0–22), 64 Conv2d modules and 57 BatchNorm2d modules. The 23 indexed modules are the main architecture blocks, not the total number of internal layers.

**Figure 2.2.** Model summary showing the layer/module listing, output shapes, parameter count and computational cost.  
[Insert SS02: screenshot of the actual `model_summary.py` output or generated layer table. Do not create or estimate missing layer-level values.]

### 2.3 Assumptions about the data

The architecture makes the following assumptions about the fabric images:

1. **Local visual patterns:** Defects such as holes, oil spots and missing yarn create local texture or appearance changes that convolutional filters can learn.
2. **Variable defect locations:** Defects may occur in different image regions. Convolutional feature extraction supports detection at different locations.
3. **Different defect sizes:** The three detection scales combine fine spatial details with higher-level context. This is useful because defect bounding boxes vary in size.
4. **Sufficient visual information:** The 640-pixel input is assumed to retain enough detail for useful detection. However, the dataset audit found small defect boxes, so resizing may remove information from the smallest defects.
5. **Manageable appearance variation:** Fabric colour, lighting and texture may vary. The colour analysis identified associations between colour characteristics and some defect classes. Augmentation and later error analysis are needed to check whether the model relies on defect features rather than colour shortcuts.

**Figure 2.3.** Representative ISL-Knit fabric images with their annotated defect bounding boxes.  
[Insert SS03: a screenshot or image panel showing several real labelled samples from the prepared dataset.]

**Figure 2.4.** Defect bounding-box size distribution used to assess the input-resolution and multi-scale assumptions.  
[Insert SS04: the Q1 defect-size / bounding-box distribution figure.]

### 2.4 Justification of architecture design decisions

| Design decision | Selected choice and justification |
|---|---|
| Depth | YOLOv8s provides a moderate depth and feature-extraction capacity while keeping training and inference costs practical for the available data and GPU. |
| Width | The small variant uses narrower channel widths than larger YOLO variants. This reduces parameters and computation while retaining three detection scales. |
| Input resolution | 640 × 640 was selected as a practical balance between computational cost and visibility of small defects. The audit indicates that some defects remain very small at this resolution. |
| Normalisation | Batch Normalisation is used in convolutional blocks to help stabilise intermediate feature distributions. The model summary identified 57 BatchNorm2d modules. |
| Activation | SiLU is used in the YOLOv8 convolutional architecture. Its smooth non-linear response supports learning complex visual patterns. |
| Regularisation | The training configuration uses weight decay of 0.0005 and augmentation including mosaic, HSV variation and horizontal flipping. No Dropout modules were identified in the model summary. |
| Output layer | Three anchor-free detection branches predict boxes and seven defect classes at strides 8, 16 and 32. This supports defects at different spatial scales. |
| Loss function | YOLOv8 combines Binary Cross-Entropy (BCE) classification loss, Complete IoU (CIoU) bounding-box loss and Distribution Focal Loss (DFL) for box-coordinate regression. These components support both defect classification and localisation. |

Most architectural components and their default settings are supplied by the Ultralytics YOLOv8 implementation. Project-specific choices include the YOLOv8s scale, seven output classes, 640-pixel input resolution, fixed seed and training configuration.

The loss and output formulation match the task because the inspection system must identify defect types and locate them on the fabric. Question 1 identified missed defects (false negatives) as more costly than unnecessary inspections (false positives). Therefore, the detection threshold will be selected using validation data, with particular attention to recall. The threshold must not be selected using the held-out test set.

**Figure 2.5.** Initial E00 training configuration showing the model, image size, batch size, optimiser, learning rate and augmentation settings.  
[Insert SS05: screenshot of the saved E00 run settings / configuration. E00 used YOLOv8s, 640-pixel images, batch size 2, AdamW, learning rate 0.001, seed 42 and a maximum of 30 epochs.]

## 2(b) Discuss Alternative Approaches

### 2.5 Why an ANN is appropriate

The ISL-Knit dataset contains 3,375 images and 20,679 annotated defect instances across seven classes. It consists of fabric images with spatially located annotations. Therefore, the task requires both visual pattern recognition and defect localisation.

A convolutional ANN can learn hierarchical visual features, from edges and texture patterns to more complex defect representations. YOLOv8s learns feature extraction and bounding-box prediction jointly. Its pretrained weights provide a useful starting point, while its relatively compact size is practical for the proposed inspection application.

### 2.6 Alternative approaches considered

| Alternative | When it would be appropriate | Why it is less suitable as the main approach for this problem |
|---|---|---|
| Image-classification CNN | When only image-level defect presence or category is required. | It does not directly locate defects, which is necessary for fabric inspection. |
| Faster R-CNN | When detection accuracy is prioritised over speed and greater computational resources are available. | Its two-stage detection process is generally more computationally demanding than a single-stage detector. |
| DETR-style detector | When a sufficiently large dataset and resources for transformer-based detection are available. | The available training set is comparatively limited, making it a less practical initial choice. |
| Autoencoder or PatchCore | When labelled defect examples are scarce or the objective is to detect previously unseen anomalies. | The dataset already provides bounding-box annotations for seven known defect classes, supporting supervised detection. |
| Tiling-based detection | When very small defects are lost after resizing high-resolution images. | Tiling adds processing and prediction-merging complexity. It can be considered if 640-pixel detection is inadequate. |
| Texture-feature SVM | When a lightweight model using manually designed texture features is desired, particularly with limited data. | Handcrafted features may represent complex spatial patterns and overlapping defects less effectively. It is retained as the classical baseline for Q3. |

For each alternative, the choice depends on the actual inspection requirements. Image-level classification is suitable when locating the defect is unnecessary. Faster R-CNN may be appropriate when additional computation is available and detection accuracy is the main priority. Anomaly detection is more appropriate when defect labels are unavailable or unknown defect types are the main concern. Tiling may be useful if the smallest defects cannot be detected at the selected input resolution.

### 2.7 Pre-result performance prediction

**This prediction was recorded before comparative model results were available and should remain unchanged after results are observed.**

Before conducting comparative experiments, I expect YOLOv8s to outperform the texture-feature SVM on the primary metric, mAP@0.5. The defects are small, numerous and sometimes overlapping. The median short-side bounding-box dimension is approximately 14.4 pixels at the selected 640-pixel resolution. A fixed-window, fixed-stride detector may therefore struggle to localise defects accurately. In comparison, YOLOv8s learns visual features and bounding-box regression jointly using pretrained weights.

The SVM may remain competitive for texture-related categories such as Hole and Contamination, but it may perform less effectively on rare classes with fewer training examples. This is a hypothesis recorded before comparative results were obtained. It will be evaluated using the same fixed data split and held-out test set.

### 2.8 Evidence and implementation notes

Architecture values and dimensions were generated using the project scripts `model_summary.py` and `make_arch_diagram.py`. The architecture configuration is YOLOv8s with seven classes and 640-pixel input size. The model summary was generated using the installed Ultralytics version 8.4.190.

The one-epoch GPU smoke test was used only to verify that the local training environment works. Its metrics are not treated as comparative coursework results. Formal model performance is reported separately in Question 3.

## Screenshot and figure checklist

| Figure / SS | Evidence to capture | Suggested filename | Placement |
|---|---|---|---|
| Figure 2.1 | YOLOv8s architecture diagram | `Q2_SS01_architecture.png` | Section 2.1 |
| Figure 2.2 | Detailed model summary / layer listing | `Q2_SS02_model_summary.png` | Section 2.2 |
| Figure 2.3 | Real fabric images with annotated defect boxes | `Q2_SS03_annotated_samples.png` | Section 2.3 |
| Figure 2.4 | Q1 defect-size / bounding-box distribution | `Q2_SS04_defect_size_eda.png` | Section 2.3 |
| Figure 2.5 | E00 saved training configuration | `Q2_SS05_training_configuration.png` | Section 2.4 |

**Submission check:** The architecture diagram from the original Q2 document is already included. Add the remaining screenshots at the marked locations. The layer table above summarises the main architectural stages using the verified dimensions available in the current Q2 material. Add the full module-by-module table only from the actual generated model-summary file. Do not estimate missing layer outputs or parameter counts. Keep the pre-result performance prediction unchanged. The one-epoch smoke test is not a comparative model-performance result.
