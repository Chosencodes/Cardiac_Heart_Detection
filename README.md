# Cardiac Heart Detection

Detects the heart location in chest X-rays using bounding box regression.

## Dataset
RSNA Pneumonia Detection Challenge chest X-rays re-labeled for cardiac detection.
496 labeled images, 400 train / 96 validation.

## Model
ResNet50 fine-tuned for bounding box regression (4 coordinate outputs).

## Results
- Mean IoU: 75.8%
- Overall MAE: 6.4 pixels on 224x224 images

## Tech Stack
- PyTorch
- PyTorch Lightning
- Albumentations
- pydicom

## How to Run
1. Download RSNA Pneumonia Detection Challenge dataset from Kaggle
2. Run preprocess.ipynb
3. Run train.ipynb
4. Run evaluate.ipynb
