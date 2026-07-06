# 01 - Image Classification with EfficientNet-B0

**Task**: Binary classification (Cat vs Dog)
**Model**: EfficientNet-B0 (pretrained on ImageNet, fine-tuned)
**Dataset**: [Cats-vs-Dogs](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset) — 25K images
**GPU time**: ~15 min on T4

## What you'll learn
- Transfer learning: use a pretrained backbone instead of training from scratch
- Freeze/unfreeze strategy: train head first, then fine-tune deeper layers
- Data augmentation: RandomHorizontalFlip, ColorJitter, RandomRotation
- Evaluation: confusion matrix, per-class precision/recall

## Run on Kaggle
```bash
python ../scripts/push_to_kaggle.py notebook.ipynb --acc T4
```
Or upload manually: Kaggle → New Notebook → Upload → Add dataset → Enable GPU T4 → Run All
