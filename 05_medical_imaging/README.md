# 05 - Medical Imaging: Pneumonia Detection from Chest X-Rays

**Task**: Binary classification (Normal vs Pneumonia)
**Models**: ResNet50 + DenseNet121 (transfer learning)
**Dataset**: [Chest X-Ray Images](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — 5,863 images
**GPU time**: ~20 min on T4

## What you'll learn
- Handling class imbalance with WeightedRandomSampler
- Medical AI metrics: sensitivity, specificity, ROC curve, AUC
- Grad-CAM: visualize which regions the model focuses on (critical for interpretability)
- Ethical considerations in medical AI (bias, human-in-the-loop, not a diagnostic tool)

## Run on Kaggle
```bash
python ../scripts/push_to_kaggle.py notebook.ipynb --acc T4
```
