#!/usr/bin/env python3
"""
Download ONNX models required by the 3DGS worker pipeline.

Usage:
    python scripts/download_models.py

Models downloaded to: ./models/
  - yolov8n.onnx                              (~12 MB)  person detection for COLMAP masking
  - depth_anything_v2_vits.onnx               (~94 MB)  monocular depth for point cloud densification
  - vocab_tree_flickr100K_words256K.bin       (~200 MB) COLMAP sequential loop closure detection
"""
import shutil
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def download_depth_anything():
    dst = MODELS_DIR / "depth_anything_v2_vits.onnx"
    if dst.exists():
        print(f"[skip] {dst.name} already exists ({dst.stat().st_size // 1024} KB)")
        return

    print("Downloading Depth Anything V2 Small ONNX from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(
            "onnx-community/depth-anything-v2-small",
            "onnx/model.onnx",
        )
        shutil.copy(p, dst)
    except ImportError:
        import urllib.request
        url = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
        urllib.request.urlretrieve(url, dst)

    print(f"[ok] {dst.name} — {dst.stat().st_size // (1024*1024)} MB")


def download_yolov8n():
    dst = MODELS_DIR / "yolov8n.onnx"
    if dst.exists():
        print(f"[skip] {dst.name} already exists ({dst.stat().st_size // 1024} KB)")
        return

    print("Exporting YOLOv8n to ONNX via ultralytics...")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ultralytics not installed — installing temporarily...")
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "ultralytics"],
            check=True,
        )
        from ultralytics import YOLO

    m = YOLO("yolov8n.pt")
    out = m.export(format="onnx", imgsz=640, dynamic=False)
    shutil.copy(out, dst)
    print(f"[ok] {dst.name} — {dst.stat().st_size // (1024*1024)} MB")


def download_vocab_tree():
    dst = MODELS_DIR / "vocab_tree_flickr100K_words256K.bin"
    if dst.exists():
        print(f"[skip] {dst.name} already exists ({dst.stat().st_size // (1024*1024)} MB)")
        return

    url = "https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin"
    print(f"Downloading COLMAP vocab tree (~200 MB) from {url} ...")

    def _progress(count, block_size, total):
        mb = count * block_size / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        print(f"\r  {mb:.0f} / {total_mb:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dst, reporthook=_progress)
    print()
    print(f"[ok] {dst.name} — {dst.stat().st_size // (1024*1024)} MB")


if __name__ == "__main__":
    print(f"Model directory: {MODELS_DIR}\n")
    download_depth_anything()
    download_yolov8n()
    download_vocab_tree()
    print("\nDone. Start services with: sudo docker compose up")
