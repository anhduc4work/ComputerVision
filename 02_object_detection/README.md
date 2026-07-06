# 02 - Object Detection with YOLOv8

**Task**: Multi-class object detection with bounding boxes
**Model**: YOLOv8-nano (3.2M params, real-time capable)
**Dataset**: [COCO 25-Class](https://www.kaggle.com/datasets/malaychand/coco-25-class-object-detection-yolo-datasets) — ~1GB
**GPU time**: ~30 min on T4

## What you'll learn
- How detection differs from classification (localization + classification)
- YOLO architecture: anchor-free, single-pass detection
- Metrics: mAP@0.5, mAP@0.5:0.95, precision, recall
- ONNX export for deployment

## Run on Kaggle
```bash
python ../scripts/push_to_kaggle.py notebook.ipynb --acc T4
```
