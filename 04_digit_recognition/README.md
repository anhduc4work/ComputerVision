# 04 - Digit Recognition: CNN vs Vision Transformer

**Task**: 10-class digit classification (MNIST)
**Models**: Custom CNN (baseline) vs ViT-Tiny (from timm)
**Dataset**: [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) competition
**GPU time**: ~10 min on T4

## What you'll learn
- Classic CNN: convolution → pooling → FC (the traditional approach)
- Vision Transformer: patch embedding → self-attention → classification
- When CNN beats ViT (small data) and when ViT wins (large data, pretraining)
- Kaggle competition submission format

## Run on Kaggle
```bash
python ../scripts/push_to_kaggle.py notebook.ipynb --acc T4
```
