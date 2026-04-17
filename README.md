# 3DGS Services

A production pipeline for converting real estate walkthrough videos into 3D Gaussian Splatting (3DGS) models, served via a REST API.

## Architecture

```
FastAPI (API)  →  Redis (broker + job store)  →  Celery (worker)
```

**API** handles video upload, job creation, and status polling.  
**Worker** runs the full reconstruction pipeline asynchronously.

## Pipeline

```
Video → Frame Extraction → Person Masking → COLMAP SfM → Depth Augmentation → OpenSplat → Postprocess
```

| Step | Description |
|---|---|
| Extract | fps-based frame extraction with blur/dedupe filtering + CLAHE |
| Mask | YOLOv8n person detection → COLMAP masks (removes ghost agents) |
| COLMAP | Sequential matching + vocab tree loop closure → sparse point cloud |
| Depth | Depth Anything V2 depth maps → augment sparse cloud |
| OpenSplat | 3DGS training on dense point cloud |
| Postprocess | Opacity filter + `.splat` export for web viewing |

## Quality Presets

| Preset | FPS | Matcher | Iterations |
|---|---|---|---|
| fast | 0.3 | exhaustive | 4,000 |
| balanced | 3.0 | sequential + loop closure | 15,000 |
| high | 1.0 | sequential + loop closure | 15,000 |
| ultra | 1.5 | sequential + loop closure | 30,000 |

## Test Result

Processed a 5-minute 22-second real estate walkthrough video at **3 FPS** in **83 minutes** using approximately **1.6 GB VRAM**, producing a 12 MB `.splat` file ready for web viewing. The result allows free navigation between rooms. Quality heavily depends on video quality and camera coverage.

## Setup

### 1. Download ML models

```bash
python scripts/download_models.py
```

Downloads to `./models/`:
- `yolov8n.onnx` — person detection (~12 MB)
- `depth_anything_v2_vits.onnx` — depth estimation (~94 MB)
- `vocab_tree_flickr100K_words256K.bin` — COLMAP loop closure (~113 MB)

### 2. Start services

```bash
sudo docker compose up
```

## API

### Upload video

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@video.mp4" \
  -F "quality=balanced"
# → {"job_id": "abc123"}
```

### Check status

```bash
curl http://localhost:8000/job/abc123
```

### Download output

```bash
curl -o model.splat http://localhost:8000/job/abc123/download/splat
```

View `.splat` files at [antimatter15.com/splat](https://antimatter15.com/splat/).

## Requirements

- NVIDIA GPU (tested on 24 GB VRAM — runs on ~1.6 GB)
- Docker + NVIDIA Container Toolkit
- CUDA 12.x
