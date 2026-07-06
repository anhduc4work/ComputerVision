# Computer Vision Research Lab

A hands-on learning repo with 5 practical CV notebooks, each designed to run on **Kaggle free GPU** (T4/P100).

## Your GPU Quota (refreshes weekly)
- **GPU**: ~30 hours/week (plenty for all 5 notebooks)
- **TPU**: ~20 hours/week

## Notebooks — Learning Path (Recommended Order)

### 1. Image Classification — EfficientNet-B0
**Dataset**: Cats-vs-Dogs (25K images) | **GPU time**: ~15 min
- Learn: Transfer learning, fine-tuning strategy (freeze then unfreeze)
- Key concepts: ImageNet pretraining, learning rate scheduling, data augmentation
- Model: EfficientNet-B0 (5.3M params — efficient yet accurate)

### 2. Object Detection — YOLOv8
**Dataset**: COCO 25-Class (~1GB) | **GPU time**: ~30 min
- Learn: How detection differs from classification (bounding boxes + classes)
- Key concepts: Anchor-free detection, mAP metric, NMS, YOLO architecture
- Model: YOLOv8-nano (3.2M params — real-time capable)

### 3. Image Segmentation — U-Net
**Dataset**: Pothole Segmentation (~240MB) | **GPU time**: ~30 min
- Learn: Pixel-level predictions, encoder-decoder architecture
- Key concepts: Dice loss, IoU metric, skip connections, albumentations
- Model: U-Net with ResNet34 encoder

### 4. Digit Recognition — CNN vs ViT
**Dataset**: MNIST (Digit Recognizer competition) | **GPU time**: ~10 min
- Learn: Compare classic CNN with modern Vision Transformer
- Key concepts: Convolutions vs self-attention, patch embeddings, when to use what
- Model: Custom CNN + ViT-Tiny

### 5. Medical Imaging — Chest X-Ray
**Dataset**: Chest X-Ray Pneumonia (5.8K images) | **GPU time**: ~20 min
- Learn: Transfer learning for medical AI, model interpretability
- Key concepts: Class imbalance handling, ROC/AUC, Grad-CAM visualization
- Model: ResNet50 + DenseNet121

## How to Run

### Option A: Push to Kaggle (recommended)
```bash
python scripts/push_to_kaggle.py notebooks/01_image_classification_efficientnet.ipynb --gpu
```

### Option B: Upload manually
1. Go to kaggle.com/code → New Notebook
2. File → Upload Notebook → select .ipynb
3. Add the dataset (listed in each notebook header)
4. Enable GPU in Settings → Accelerator
5. Run All

## Key Libraries
| Library | Purpose |
|---------|---------|
| PyTorch | Deep learning framework |
| timm | Pretrained models (EfficientNet, ViT, etc.) |
| ultralytics | YOLOv8 |
| segmentation_models_pytorch | U-Net, DeepLabV3 |
| albumentations | Fast image augmentation |
| matplotlib/seaborn | Visualization |
