# Cell Segmentation with Vision Transformers

Deep learning pipeline for blood cell segmentation using a pretrained Vision Transformer encoder and boundary-aware multi-task learning.

This project investigates a common microscopy failure mode:

**High Dice does NOT guarantee correct cell counts**

Semantic segmentation models often merge adjacent cells.
Instance separation can be improved using an auxiliary boundary prediction head.

---

## Problem

In microscopy analysis, downstream tasks require object-level correctness:

* Cell counting
* Morphology analysis
* Phenotype classification

However, conventional segmentation models optimize overlap metrics (Dice/IoU), allowing merged cells while still scoring well.

---

## Key Idea

Train the model to predict **regions AND borders**.

Multi-task objective:

* Head A — Cell foreground mask
* Head B — Boundary map

Boundary supervision sharpens edges and improves connected-component separation.

---

## Dataset Analysis

Findings:

* Train and test distributions aligned
* Cell area ≈ 6,000–14,000 px²
* Approximate size ≈ 100×100 px
* Same magnification across slides (some with scale bar)

Observation:
Watershed splitting fails because of thin boundaries + crowded objects

---

## Training Strategy

### Data Split

* 75% training
* 25% validation
* Image-level split to prevent leakage

### Patch Sampling

Cells ≈ 100 px wide:

* Sample one **224×224 patch per image per epoch**
* Crop mask identically

### Preprocessing

* Standard normalization
* Augmentation during training only

---

## Model Architecture

### Encoder

DINOv2 ViT-L/14 pretrained backbone
(UNI pathology foundation model considered, but access was blocked until yesterday afternoon)

### Decoder

Stepwise upsampling segmentation decoder

Two variants trained:

1. Plain segmentation model
2. Segmentation + auxiliary boundary head

### Loss Functions

Plain model

```
BCE + Dice
```

Boundary model

```
BCE + Dice (mask)
+ BCE (boundary)
```

Goal: improve instance separation rather than semantic overlap.

---

## Evaluation

### Inference Pipeline

1. Tile full validation images
2. Predict patch probabilities
3. Stitch to full resolution
4. Apply threshold
5. Connected-component labelling

### Metrics

Pixel-level:

* Dice
* IoU

Instance-level:

* MAE (cell count error)
* F1
* mAP

---

## Results

### Quantitative Performance

| Metric | 0.5 Threshold | 0.95 Threshold |
| ------ | ------------- | -------------- |
| Dice   | 0.9517        | 0.8995         |
| IoU    | 0.9084        | 0.8185         |
| F1     | 0.5458        | 0.76461        |

### Key Observations from Validation Results

* Pixel metrics peak at moderate thresholds (0.4–0.6)
* Instance metrics improve at high thresholds (~0.95)
* Boundary head reduces counting error without hurting Dice

**Conclusion:** Most errors occur at instance separation, not classification.

Threshold need to be adjusted based on the downstream tasks (confluence map or cell counting)

---

## Qualitative Results
### Bad Segmentation

<img width="1137" height="788" alt="image" src="https://github.com/user-attachments/assets/ff7739eb-1bad-43be-8464-e6557c19f7bd" />


### t=0.5

<img width="1389" height="824" alt="image" src="https://github.com/user-attachments/assets/52b8dfa0-f4a6-4c80-8d61-3525bc8ef2d0" />

### t=0.95

<img width="1389" height="824" alt="image" src="https://github.com/user-attachments/assets/1e164193-e8c3-4f57-a96b-26283d41d92c" />

---


## Limitations & Future Work

Microscopy pipelines often rely on semantic masks for quantification, but biological analysis requires object-level correctness.

Boundary supervision bridges semantic segmentation and instance segmentation without detection architectures.

Future works: 

* Distance transform / surface loss for boundary head

* Pathology foundation models (UNI) for (potentially) better pretrained weights

* Longer training with LR scheduling

Observation: Instance-level difficulty dominates performance more than semantic segmentation quality.
