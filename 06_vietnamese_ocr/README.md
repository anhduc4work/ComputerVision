# 06 - Vietnamese OCR: Text Detection & Recognition

**Task**: Detect and recognize Vietnamese text in images
**Models**: PaddleOCR (detection + recognition) + VietOCR (Vietnamese-specific recognition)
**Dataset**: [Vietnamese-OCR](https://www.kaggle.com/datasets/trongnguyen04/vietnamese-ocr) — 1GB
**GPU time**: ~15 min on T4

## What you'll learn
- OCR pipeline: text detection (find text regions) → text recognition (read text)
- Vietnamese-specific challenges: diacritics (dấu), tonal marks, Unicode handling
- PaddleOCR: all-in-one solution with Vietnamese language support
- VietOCR: purpose-built for Vietnamese, uses Transformer decoder
- Evaluation metrics: Character Error Rate (CER), Word Error Rate (WER)
- Practical document OCR pipeline

## Key Models
| Model | Strengths | Architecture |
|-------|-----------|-------------|
| PaddleOCR | All-in-one, fast, multi-language | DB (detection) + SVTR (recognition) |
| VietOCR | Best Vietnamese accuracy | CNN encoder + Transformer decoder |
