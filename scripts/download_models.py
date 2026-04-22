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
import os
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


def download_viewcrafter():
    """Download ViewCrafter_25_512 + DUSt3R weights (~6 GB total). Opt-in.

    Skipped unless env VIEWCRAFTER_DOWNLOAD=1 is set, since weights are large
    and the ViewCrafter step is disabled by default in settings.yaml.
    """
    if os.environ.get("VIEWCRAFTER_DOWNLOAD", "0").strip().lower() not in ("1", "true", "yes"):
        print("[skip] ViewCrafter weights (set VIEWCRAFTER_DOWNLOAD=1 to download)")
        return

    vc_dir = MODELS_DIR / "viewcrafter"
    vc_dir.mkdir(exist_ok=True)

    dust3r = vc_dir / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    if dust3r.exists():
        print(f"[skip] {dust3r.name} already exists")
    else:
        url = "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        print(f"Downloading DUSt3R from {url} ...")
        urllib.request.urlretrieve(url, dust3r)
        print(f"[ok] {dust3r.name} — {dust3r.stat().st_size // (1024*1024)} MB")

    ckpt = vc_dir / "ViewCrafter_25_512.ckpt"
    if ckpt.exists():
        print(f"[skip] {ckpt.name} already exists")
        return

    print("Downloading ViewCrafter_25_512 from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("Drexubery/ViewCrafter_25_512", "model.ckpt")
        shutil.copy(p, ckpt)
    except ImportError:
        print("  huggingface_hub not installed — install it or download manually")
        return
    except Exception as exc:
        print(f"  ViewCrafter download failed: {exc}")
        return
    print(f"[ok] {ckpt.name} — {ckpt.stat().st_size // (1024*1024)} MB")


if __name__ == "__main__":
    print(f"Model directory: {MODELS_DIR}\n")
    download_depth_anything()
    download_yolov8n()
    download_vocab_tree()
    download_viewcrafter()
    print("\nDone. Start services with: sudo docker compose up")
