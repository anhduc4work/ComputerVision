"""
Utility script to push notebooks to Kaggle using the Kaggle API.
Usage: python push_to_kaggle.py <notebook_path> [--accelerator NvidiaTeslaT4]
       python push_to_kaggle.py --all --accelerator NvidiaTeslaT4
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Valid accelerator values for Kaggle machine_shape field
ACCELERATORS = {
    "P100": "NvidiaTeslaP100",
    "T4": "NvidiaTeslaT4",
    "T4Highmem": "NvidiaTeslaT4Highmem",
    "A100": "NvidiaTeslaA100",
    "L4": "NvidiaL4",
    "L4X1": "NvidiaL4X1",
    "H100": "NvidiaH100",
    "RTX6000": "NvidiaRtxPro6000",
    "TPUv3": "TpuV38",
    "TPUv5e": "TpuV5E8",
    "TPUv6e": "TpuV6E8",
}

NOTEBOOK_CONFIGS = {
    "01_image_classification_efficientnet": {
        "title": "CV Lab - Image Classification EfficientNet-B0",
        "dataset_sources": ["shaunthesheep/microsoft-catsvsdogs-dataset"],
        "competition_sources": [],
        "keywords": ["computer vision", "image classification", "transfer learning", "efficientnet"],
    },
    "02_object_detection_yolov8": {
        "title": "CV Lab - Object Detection YOLOv8",
        "dataset_sources": ["malaychand/coco-25-class-object-detection-yolo-datasets"],
        "competition_sources": [],
        "keywords": ["computer vision", "object detection", "yolo", "yolov8"],
    },
    "03_image_segmentation_unet": {
        "title": "CV Lab - Image Segmentation U-Net",
        "dataset_sources": ["raunakkesharwani/pothole-image-segmentation-dataset"],
        "competition_sources": [],
        "keywords": ["computer vision", "image segmentation", "unet", "pytorch"],
    },
    "04_digit_recognition_vit": {
        "title": "CV Lab - Digit Recognition CNN vs ViT",
        "dataset_sources": [],
        "competition_sources": ["digit-recognizer"],
        "keywords": ["computer vision", "vision transformer", "cnn", "mnist"],
    },
    "05_medical_imaging_chest_xray": {
        "title": "CV Lab - Medical Imaging Chest X-Ray",
        "dataset_sources": ["paultimothymooney/chest-xray-pneumonia"],
        "competition_sources": [],
        "keywords": ["computer vision", "medical imaging", "transfer learning", "grad-cam"],
    },
}


def get_kaggle_username():
    """Get Kaggle username from kaggle.json or environment."""
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        with open(kaggle_json) as f:
            return json.load(f).get("username", "your-username")
    return "your-username"


def resolve_accelerator(value: str) -> str:
    """Resolve shorthand accelerator name to Kaggle machine_shape value."""
    if not value:
        return ""
    # Already a full Kaggle value
    if value.startswith("Nvidia") or value.startswith("Tpu"):
        return value
    # Shorthand lookup
    if value in ACCELERATORS:
        return ACCELERATORS[value]
    # Case-insensitive search
    for k, v in ACCELERATORS.items():
        if k.lower() == value.lower():
            return v
    print(f"Unknown accelerator: {value}")
    print(f"Valid values: {list(ACCELERATORS.keys())} or full Kaggle IDs")
    sys.exit(1)


def create_kernel_metadata(notebook_path: Path, machine_shape: str, enable_internet: bool):
    """Create kernel-metadata.json for Kaggle push."""
    stem = notebook_path.stem
    config = NOTEBOOK_CONFIGS.get(stem)
    # If file is named notebook.ipynb, use parent folder name as key
    if not config and stem == "notebook":
        folder_name = notebook_path.parent.name
        config = NOTEBOOK_CONFIGS.get(folder_name)
    if not config:
        print(f"Unknown notebook: {stem} (folder: {notebook_path.parent.name})")
        print(f"Known notebooks: {list(NOTEBOOK_CONFIGS.keys())}")
        sys.exit(1)

    username = get_kaggle_username()
    slug = stem.replace("_", "-")

    metadata = {
        "id": f"{username}/{slug}",
        "title": config["title"],
        "code_file": str(notebook_path.name),
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true" if machine_shape else "false",
        "enable_internet": "true" if enable_internet else "false",
        "machine_shape": machine_shape,
        "dataset_sources": config["dataset_sources"],
        "competition_sources": config["competition_sources"],
        "kernel_sources": [],
        "model_sources": [],
        "keywords": config["keywords"],
    }

    metadata_path = notebook_path.parent / "kernel-metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Created metadata: {metadata_path}")
    print(f"  Accelerator: {machine_shape or 'CPU only'}")
    return metadata_path


def push_notebook(notebook_path: Path, machine_shape: str, enable_internet: bool):
    """Push notebook to Kaggle."""
    metadata_path = create_kernel_metadata(notebook_path, machine_shape, enable_internet)

    cmd = ["kaggle", "kernels", "push", "-p", str(notebook_path.parent)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Successfully pushed: {notebook_path.name}")
        print(result.stdout)
    else:
        print(f"Error pushing notebook: {result.stderr}")
        sys.exit(1)

    # Clean up metadata
    metadata_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Push CV notebooks to Kaggle",
        epilog=f"Accelerator shortcuts: {', '.join(ACCELERATORS.keys())}",
    )
    parser.add_argument("notebook", nargs="?", help="Path to .ipynb file")
    parser.add_argument(
        "--accelerator", "--acc", default="T4",
        help="GPU/TPU type (default: T4). Shortcuts: P100, T4, T4Highmem, A100, L4, H100, TPUv3, TPUv5e",
    )
    parser.add_argument("--no-internet", action="store_true", help="Disable internet access")
    parser.add_argument("--all", action="store_true", help="Push all notebooks")
    args = parser.parse_args()

    machine_shape = resolve_accelerator(args.accelerator)
    enable_internet = not args.no_internet

    if args.all:
        repo_root = Path(__file__).parent.parent
        nbs = sorted(repo_root.glob("*/notebook.ipynb"))
        if not nbs:
            nbs = sorted(repo_root.glob("notebooks/*.ipynb"))
        for nb in nbs:
            push_notebook(nb, machine_shape, enable_internet)
    elif args.notebook:
        notebook_path = Path(args.notebook).resolve()
        if not notebook_path.exists():
            print(f"File not found: {notebook_path}")
            sys.exit(1)
        push_notebook(notebook_path, machine_shape, enable_internet)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
