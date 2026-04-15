"""Frame extraction with blur/dedupe filtering — adapted from tmp_process_video.py."""
import logging
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .context import PipelineContext, ProgressCallback

log = logging.getLogger(__name__)

QUALITY_PRESETS: dict = {
    "fast": {
        "target_frames": 120,
        "max_image_size": 1280,
        "max_extract_fps": 8,
        "blur_threshold": 40.0,
        "dedupe_threshold": 2.5,
        "iterations": 4000,
    },
    "balanced": {
        "target_frames": 220,
        "max_image_size": 1600,
        "max_extract_fps": 10,
        "blur_threshold": 30.0,
        "dedupe_threshold": 2.0,
        "iterations": 7000,
    },
    "high": {
        "target_frames": 400,
        "max_image_size": 2048,
        "max_extract_fps": 15,
        "blur_threshold": 20.0,
        "dedupe_threshold": 1.5,
        "iterations": 15000,
    },
    "ultra": {
        "target_frames": 512,
        "max_image_size": 2048,
        "max_extract_fps": 20,
        "blur_threshold": 15.0,
        "dedupe_threshold": 1.0,
        "iterations": 30000,
    },
}


def get_quality_params(preset: str) -> dict:
    return dict(QUALITY_PRESETS.get(preset.strip().lower(), QUALITY_PRESETS["balanced"]))


def get_video_duration(video_path: Path) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True,
        )
        return float(result.stdout.strip()) if result.returncode == 0 else 0.0
    except Exception:
        return 0.0


def _extract_raw(video_path: Path, raw_dir: Path, fps: int, max_size: Optional[int]) -> int:
    raw_dir.mkdir(parents=True, exist_ok=True)
    vf = [f"fps={fps}"]
    if max_size:
        vf.append(
            f"scale='if(gt(iw,ih),min(iw,{max_size}),-2)':'if(gt(iw,ih),-2,min(ih,{max_size}))'"
        )
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", ",".join(vf), "-q:v", "2",
        str(raw_dir / "frame_%04d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr}")
    return len(list(raw_dir.glob("*.jpg")))


def _select_frames(
    raw_dir: Path,
    out_dir: Path,
    target: int,
    blur_thresh: Optional[float],
    dedupe_thresh: Optional[float],
) -> int:
    raw_frames = sorted(raw_dir.glob("frame_*.jpg"))
    if not raw_frames:
        return 0

    def _filter(bt, dt):
        selected, prev_thumb = [], None
        for fp in raw_frames:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if bt is not None and cv2.Laplacian(gray, cv2.CV_64F).var() < bt:
                continue
            if dt is not None:
                thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
                if prev_thumb is not None:
                    diff = float(np.mean(np.abs(thumb.astype(np.float32) - prev_thumb.astype(np.float32))))
                    if diff < dt:
                        continue
                prev_thumb = thumb
            selected.append(fp)
        return selected

    min_req = min(len(raw_frames), max(30, int(target * 0.10)))
    selected = _filter(blur_thresh, dedupe_thresh)

    if len(selected) < min_req:
        log.warning("Filtering left only %d frames, relaxing blur threshold", len(selected))
        selected = _filter(blur_thresh * 0.6 if blur_thresh else None, dedupe_thresh)
    if len(selected) < min_req:
        selected = _filter(None, dedupe_thresh)
    if len(selected) < min_req:
        log.warning("Falling back to unfiltered frames")
        selected = list(raw_frames)

    # Uniform downsample to target
    if len(selected) > target:
        step = len(selected) / target
        selected = [selected[int(i * step)] for i in range(target)]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for idx, src in enumerate(selected):
        dst = out_dir / f"frame_{idx + 1:04d}.jpg"
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    return len(selected)


def run(ctx: PipelineContext, params: dict, on_progress: ProgressCallback) -> int:
    on_progress("extract", 5, "Extracting frames from video...")

    duration = get_video_duration(ctx.video_path)
    target = int(params.get("target_frames") or 220)
    max_fps = int(params.get("max_extract_fps") or 10)
    max_size = int(params.get("max_image_size") or 0) or None

    fps = 3
    if target and duration > 0:
        fps = max(2, min(max_fps, int(math.ceil(target / max(duration, 1.0) * 1.5))))
    else:
        fps = max(2, max_fps)

    raw_dir = ctx.frames_dir.parent / "frames_raw"
    raw_count = _extract_raw(ctx.video_path, raw_dir, fps, max_size)
    if raw_count == 0:
        raise RuntimeError("No frames extracted from video")

    count = _select_frames(
        raw_dir, ctx.frames_dir, target,
        blur_thresh=params.get("blur_threshold"),
        dedupe_thresh=params.get("dedupe_threshold"),
    )
    shutil.rmtree(raw_dir, ignore_errors=True)

    if count < 10:
        raise RuntimeError(f"Not enough usable frames ({count}). Need at least 10.")

    on_progress("extract", 15, f"Extracted {count} frames")
    log.info("[%s] Extracted %d frames", ctx.job_id, count)
    return count
