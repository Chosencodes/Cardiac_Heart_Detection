# Cardiac Heart Detection using ResNet50

A deep learning model that detects the heart location in chest X-rays using bounding box regression, with cardiothoracic ratio calculation for clinical relevance.

🔴 Live Demo: https://huggingface.co/spaces/Chosencodes/cardiac-heart-detection

## Overview

This project trains a ResNet50 model to localize the heart in frontal chest X-rays by predicting a bounding box around the cardiac region. The cardiothoracic ratio (CT ratio) is then calculated from the predicted box — a clinically used metric for detecting cardiomegaly (enlarged heart).

## Dataset

- **Source:** RSNA Pneumonia Detection Challenge (Kaggle)
- **Re-labeled:** 496 chest X-rays labeled with cardiac bounding boxes
- **Split:** 400 training / 96 validation
- **Format:** DICOM (.dcm) converted to NumPy arrays (.npy)

## Model Architecture

- **Base model:** ResNet50 pretrained on ImageNet
- **Input:** Single-channel (grayscale) chest X-ray — 224x224
- **Output:** 4 bounding box coordinates [x_min, y_min, x_max, y_max]
- **Loss function:** MSELoss (regression)
- **Optimizer:** Adam (lr=1e-4)
- **Framework:** PyTorch + PyTorch Lightning

## Results

| Metric | Value |
|--------|-------|
| Mean IoU | 81.2% |
| Overall MAE | 4.76 pixels |
| Mean CT Ratio | 0.358 (normal range) |
| High Confidence Samples (IoU > 0.8) | 60 / 96 |
| Low Confidence Samples (IoU < 0.5) | 1 / 96 |

## Clinical Features

- **Heart localization** — draws bounding box around the heart
- **Cardiothoracic ratio** — heart width divided by chest width
- **Cardiomegaly detection** — flags CT ratio > 0.5 as possibly enlarged


## Note: Model is optimized for RSNA-style frontal chest X-rays.
Best results with standardized PA/AP view radiographs.
