# 03 - Image Segmentation with U-Net

**Task**: Pixel-level binary segmentation (pothole vs background)
**Model**: U-Net with ResNet34 encoder (pretrained ImageNet)
**Dataset**: [Pothole Segmentation](https://www.kaggle.com/datasets/raunakkesharwani/pothole-image-segmentation-dataset) — 240MB
**GPU time**: ~30 min on T4

## What you'll learn
- Encoder-decoder architecture with skip connections
- Loss functions: Dice Loss + BCE (why Dice is better than plain BCE for segmentation)
- IoU (Intersection over Union) metric
- Albumentations for spatial augmentation (flips, rotations applied to both image AND mask)

## Run on Kaggle
```bash
python ../scripts/push_to_kaggle.py notebook.ipynb --acc T4
```
