# DermaScan — AI-Based Skin Cancer Detection

An interpretable, lightweight deep learning system for melanoma detection, engineered for real-world clinical deployment.

**Author:** Vishal Domale | M.Sc. IT (Artificial Intelligence), University of Mumbai  
**Guide:** Ms. Pooja Tupe

---

## Overview

DermaScan classifies dermoscopy images as **melanoma** or **non-melanoma** using an ensemble of two convolutional neural networks augmented with spatial attention and Explainable AI (Grad-CAM). The system is designed to prioritize sensitivity — catching as many true melanoma cases as possible — while remaining lightweight enough for real-world deployment.

---

## Models

| Model | Backbone | Input Size | Parameters |
|---|---|---|---|
| LesionFocusedEfficientNet | EfficientNetV2-M (ImageNet-21K) | 480×480 | 53.6M |
| LesionFocusedConvNeXt | ConvNeXt-Base (ImageNet-1K) | 288×288 | 88.1M |

Both models share a custom **Spatial Attention Gate** and a two-stage training strategy (frozen backbone → full fine-tuning).

---

## Key Results (HAM10000 validation set, 1,494 images)

| Metric | EfficientNetV2-M | ConvNeXt-Base | Ensemble |
|---|---|---|---|
| AUROC (with TTA) | 0.9415 | 0.9410 | **0.9510** |
| Melanoma Sensitivity | 94.3% | 92.7% | 88.6% |
| Overall Accuracy | 83% | 83% | **88.8%** |
| NPV (Youden threshold) | — | — | 98.9% |

On the external ISIC 2016 dataset, the ensemble achieved 77.3% accuracy, demonstrating reasonable generalization despite domain shift.

---

## Architecture Highlights

- **Spatial Attention Gate** — learns a per-pixel attention map to suppress background noise and focus on lesion regions
- **Background Suppression Loss** — composite loss (BCE + λ·BG penalty, λ=0.1) penalizes attention falling outside the lesion
- **Lesion-Aware CutMix** — uses Otsu segmentation masks to transplant only lesion regions during training, preventing shortcut learning
- **Test-Time Augmentation (TTA)** — 9-round TTA (flips, rotations, shifts) averaged for final predictions
- **Grad-CAM** — every prediction includes heatmap overlays from both models for clinical interpretability

---

## Tech Stack

| Component | Specification |
|---|---|
| Language | Python 3.12 |
| Framework | PyTorch 2.10 + CUDA 12.8 |
| Models | timm 1.0.26 |
| Augmentation | Albumentations |
| Web Interface | Gradio 5.50 |
| Platform | Google Colab Pro (NVIDIA L4, 24 GB VRAM) |

---

## Dataset

**HAM10000** — 10,015 dermoscopy images across 7 diagnostic categories (publicly available via the ISIC archive). Binary labels: `melanoma = 1`, all others `= 0`. Split: 8,017 train / 1,494 validation, deduplicated by `lesion_id` to prevent data leakage.

---

## Limitations

- Trained on a single-institution dataset with predominantly lighter skin tones; performance on darker skin tones is not validated
- Binary classification only (melanoma vs. rest); full 7-class support is future work
- Precision at the Youden threshold is ~0.42, meaning roughly half of positive predictions are false alarms (acceptable for a screening tool)

---

## Future Work

- Multi-class extension to all 7 HAM10000 categories
- Training on diverse datasets (ISIC 2019/2020, Fitzpatrick17k) for better generalization
- Model distillation and quantization for mobile/edge deployment
- Integration of patient metadata (age, sex, anatomical site) as auxiliary inputs

---

## Publication

A research paper based on this project was published in **IJRMPS, Volume 14 Issue 1 (Jan–Feb 2026)**:  
*"Engineering Considerations for Real-World Deployment of AI-Based Skin Cancer Detection Software"* — Vishal Domale, Pooja Tupe.
