# Computer Vision Research Lab

A hands-on learning repo with 5 practical CV projects, each with a notebook ready to run on **Kaggle free GPU** (T4/P100).

## Project Structure

```
ComputerVision/
├── 01_image_classification/   # EfficientNet-B0 on Cats-vs-Dogs
│   ├── README.md
│   └── notebook.ipynb
├── 02_object_detection/       # YOLOv8 on COCO 25-Class
│   ├── README.md
│   └── notebook.ipynb
├── 03_image_segmentation/     # U-Net on Pothole Detection
│   ├── README.md
│   └── notebook.ipynb
├── 04_digit_recognition/      # CNN vs ViT on MNIST
│   ├── README.md
│   └── notebook.ipynb
├── 05_medical_imaging/        # Chest X-Ray Pneumonia + Grad-CAM
│   ├── README.md
│   └── notebook.ipynb
└── scripts/
    └── push_to_kaggle.py      # CLI to push & run notebooks on Kaggle
```

## GPU Quota (refreshes weekly)
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

## How to Run — Fully Automated via MCP

This repo uses the **Kaggle MCP server** to push and run notebooks directly from Claude Code — no manual steps, no browser needed. The agent handles everything:

1. Reads the notebook `.ipynb` file
2. Pushes it to Kaggle via `mcp__kaggle__save_notebook`
3. Selects GPU type (T4 by default) via `machineShape`
4. Attaches the correct dataset/competition sources
5. Runs all cells top-to-bottom (`SaveAndRunAll`)
6. Monitors status via `mcp__kaggle__get_notebook_session_status`

### MCP Setup (`.mcp.json` at repo root)
```json
{
  "mcpServers": {
    "kaggle": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://www.kaggle.com/mcp",
               "--header", "Authorization: Bearer YOUR_KAGGLE_TOKEN"],
      "timeout": 30
    }
  }
}
```

### Key MCP Parameters for `save_notebook`
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `machineShape` | `NvidiaTeslaT4` | Select GPU type |
| `enableGpu` | `true` | Enable GPU |
| `enableInternet` | `true` | Allow pip installs |
| `kernelExecutionType` | `SaveAndRunAll` | Auto-run all cells |
| `datasetDataSources` | `["owner/dataset"]` | Attach datasets |
| `competitionDataSources` | `["competition-slug"]` | Attach competition data |

### GPU Selection

| `machineShape` value     | GPU/Accelerator            | Free tier |
|--------------------------|----------------------------|-----------|
| `NvidiaTeslaP100`        | Tesla P100 (legacy default)| Yes       |
| `NvidiaTeslaT4`          | Tesla T4 x2 (recommended) | Yes       |
| `NvidiaTeslaT4Highmem`   | Tesla T4 high memory       | Yes       |
| `NvidiaTeslaA100`        | Tesla A100                 | No        |
| `NvidiaL4`               | NVIDIA L4                  | Limited   |
| `NvidiaH100`             | NVIDIA H100                | No        |
| `NvidiaRtxPro6000`       | RTX Pro 6000               | No        |
| `TpuV38`                 | TPU v3-8                   | Yes       |
| `TpuV5E8`                | TPU v5e-8                  | Yes       |
| `TpuV6E8`                | TPU v6e-8                  | Yes       |

**Important**: Setting only `enableGpu: true` without `machineShape` silently defaults to P100. Always specify `machineShape` explicitly.

### Fallback: Manual Upload
1. Go to kaggle.com/code → New Notebook
2. File → Upload Notebook → select `.ipynb` from any project folder
3. Add the dataset (listed in each notebook header)
4. Settings → Accelerator → GPU T4 x2
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
