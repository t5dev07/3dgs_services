
==========
== CUDA ==
==========

CUDA Version 12.1.0

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
   Use the NVIDIA Container Toolkit to start this container with GPU support; see
   https://docs.nvidia.com/datacenter/cloud-native/ .

*************************
** DEPRECATION NOTICE! **
*************************
THIS IMAGE IS DEPRECATED and is scheduled for DELETION.
    https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/support-policy.md

"""
Video to 3D Gaussian Splatting Pipeline

Steps:
1. Download video from URL
2. Extract frames with FFmpeg (3 FPS)
3. Run COLMAP for Structure-from-Motion
4. Run OpenSplat for Gaussian Splatting training
5. Upload result to S3/MinIO (optional)
"""

import os
import math
import subprocess
import shutil
import tempfile
import requests
import re
import uuid
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Optional, Dict, Any

import numpy as np
import cv2


QUALITY_PRESETS: Dict[str, Dict[str, Any]] = {
    # Fast: cheapest + most robust for weak devices / quick previews
    "fast": {
        "target_frames": 120,
        "max_image_size": 1280,
        "max_extract_fps": 8,
        "blur_threshold": 40.0,  # Laplacian variance (higher = stricter)
        "dedupe_threshold": 2.5,  # mean abs diff of 32x32 grayscale thumbnails (0..255)
        "colmap_matcher": "sequential",
        "sequential_overlap": 10,
        "loop_closure": False,
        "max_num_features": 4096,
        "guided_matching": False,
        "iterations": 4000,
    },
    # Balanced: good quality at reasonable cost
    "balanced": {
        "target_frames": 220,
        "max_image_size": 1600,
        "max_extract_fps": 10,
        "blur_threshold": 30.0,
        "dedupe_threshold": 2.0,
        "colmap_matcher": "sequential",
        "sequential_overlap": 12,
        # Loop-closure requires a COLMAP vocab tree file. Many container builds don’t ship it.
        # We keep the runtime fallback (disable loop-closure on failure), but default to off for reliability.
        "loop_closure": False,
        "max_num_features": 8192,
        "guided_matching": True,
        "iterations": 7000,
    },
    # High: best quality (more time/cost)
    "high": {
        "target_frames": 350,
        "max_image_size": 2048,
        "max_extract_fps": 12,
        "blur_threshold": 20.0,
        "dedupe_threshold": 1.5,
        "colmap_matcher": "sequential",
        "sequential_overlap": 15,
        # Loop-closure requires a COLMAP vocab tree file; default off for container reliability.
        "loop_closure": False,
        "max_num_features": 12000,
        "guided_matching": True,
        "iterations": 15000,
    },
}

def _env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def get_max_input_video_bytes() -> int:
    """
    Hard limit for input video size to avoid accidental huge uploads and runaway costs.
    Configure via env:
      - MAX_INPUT_VIDEO_BYTES (highest priority)
      - MAX_INPUT_VIDEO_MB (default: 200)
    Set to 0 to disable (not recommended).
    """
    direct = _env_int("MAX_INPUT_VIDEO_BYTES")
    if direct is not None:
        return max(0, direct)
    mb = _env_int("MAX_INPUT_VIDEO_MB")
    if mb is None:
        mb = 500
    return max(0, mb) * 1024 * 1024


def get_max_input_video_seconds() -> int:
    """
    Hard limit for input duration. Configure via env:
      - MAX_INPUT_VIDEO_SECONDS (default: 120)
    Set to 0 to disable (not recommended).
    """
    sec = _env_int("MAX_INPUT_VIDEO_SECONDS")
    if sec is None:
        sec = 300
    return max(0, sec)


class InputRejectedError(RuntimeError):
    """Raised when the input video violates hard limits (size/duration)."""


def get_quality_params(preset: str) -> Dict[str, Any]:
    preset_key = (preset or "").strip().lower()
    if preset_key not in QUALITY_PRESETS:
        preset_key = "balanced"
    return dict(QUALITY_PRESETS[preset_key])


def download_video(url: str, output_path: str) -> bool:
    """Download video from URL with validation."""
    print(f"[1/5] Downloading video from {url}")
    try:
        # Use longer timeout and verify full download
        response = requests.get(url, stream=True, timeout=600)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_error:
            status = getattr(response, "status_code", "?")
            print(f"  -> ERROR: HTTP {status} ({http_error})")
            # MinIO often provides detailed error info via headers/body.
            error_code = response.headers.get("x-minio-error-code") or response.headers.get("x-amz-error-code")
            error_desc = response.headers.get("x-minio-error-desc") or response.headers.get("x-amz-error-message")
            if error_code or error_desc:
                print(f"  -> Remote error: {error_code or '-'} | {error_desc or '-'}")
            try:
                preview = (response.text or "").strip()
                if preview:
                    print(f"  -> Response body (preview): {preview[:400]}")
            except Exception:
                pass
            return False

        # Get expected size from Content-Length header
        max_bytes = get_max_input_video_bytes()
        expected_size = int(response.headers.get('Content-Length', 0))
        print(f"  -> Expected size: {expected_size / (1024*1024):.1f} MB")
        if max_bytes > 0 and expected_size > max_bytes:
            raise InputRejectedError(
                "Input video too large "
                f"({expected_size / (1024*1024):.1f} MB). "
                f"Maximum: {max_bytes / (1024*1024):.1f} MB."
            )

        # Download with progress tracking
        downloaded = 0
        too_large = False
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):  # 64KB chunks
                if chunk:
                    if max_bytes > 0 and downloaded + len(chunk) > max_bytes:
                        print(
                            "  -> ERROR: Input video exceeded maximum size while downloading "
                            f"(>{max_bytes / (1024*1024):.1f} MB). Aborting."
                        )
                        too_large = True
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

        if too_large:
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception:
                pass
            raise InputRejectedError(
                "Input video too large. "
                f"Maximum: {max_bytes / (1024*1024):.1f} MB."
            )

        # Verify file size
        actual_size = os.path.getsize(output_path)
        print(f"  -> Downloaded: {actual_size / (1024*1024):.1f} MB")

        if expected_size > 0 and actual_size < expected_size * 0.99:
            print(f"  -> WARNING: File may be incomplete! Expected {expected_size}, got {actual_size}")
            return False

        # Verify file is not empty or too small
        if actual_size < 1000:
            print(f"  -> ERROR: Downloaded file too small ({actual_size} bytes)")
            return False

        print(f"  -> Download complete: {output_path}")
        return True
    except InputRejectedError:
        raise
    except requests.exceptions.Timeout:
        print(f"  -> ERROR: Download timed out after 600s")
        return False
    except Exception as e:
        print(f"  -> ERROR: {e}")
        return False


def extract_s3_key_from_url(video_url: str, bucket: str) -> Optional[str]:
    """Best-effort: extract S3 object key from a path-style URL: /<bucket>/<key>"""
    if not video_url or not bucket:
        return None
    try:
        from urllib.parse import urlparse, unquote

        parsed = urlparse(video_url)
        raw_path = unquote(parsed.path or "")
        path = raw_path.lstrip("/")
        prefix = f"{bucket}/"
        if path.startswith(prefix):
            return path[len(prefix):]
    except Exception:
        return None
    return None


def download_video_from_s3(video_key: str, output_path: str, config: Dict[str, str]) -> bool:
    """Download video from S3/MinIO using credentials (more robust than presigned URLs)."""
    print(f"[1/5] Downloading video from S3 key: {video_key}")
    try:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=config["endpoint"],
            aws_access_key_id=config["access_key"],
            aws_secret_access_key=config["secret_key"],
            region_name=config.get("region", "us-east-1"),
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )

        expected_size = 0
        try:
            head = client.head_object(Bucket=config["bucket"], Key=video_key)
            expected_size = int(head.get("ContentLength", 0) or 0)
        except Exception as head_error:
            print(f"  -> WARNING: HeadObject failed: {head_error}")

        if expected_size:
            print(f"  -> Expected size: {expected_size / (1024*1024):.1f} MB")
        max_bytes = get_max_input_video_bytes()
        if max_bytes > 0 and expected_size and expected_size > max_bytes:
            raise InputRejectedError(
                "Input video too large "
                f"({expected_size / (1024*1024):.1f} MB). "
                f"Maximum: {max_bytes / (1024*1024):.1f} MB."
            )

        obj = client.get_object(Bucket=config["bucket"], Key=video_key)
        body = obj.get("Body")
        if body is None:
            print("  -> ERROR: S3 GetObject returned no body")
            return False

        downloaded = 0
        too_large = False
        with open(output_path, "wb") as f:
            while True:
                chunk = body.read(65536)
                if not chunk:
                    break
                downloaded += len(chunk)
                if max_bytes > 0 and downloaded > max_bytes:
                    print(
                        "  -> ERROR: Input video exceeded maximum size while downloading "
                        f"(>{max_bytes / (1024*1024):.1f} MB). Aborting."
                    )
                    too_large = True
                    break
                f.write(chunk)

        try:
            body.close()
        except Exception:
            pass

        if too_large:
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception:
                pass
            raise InputRejectedError(
                "Input video too large. "
                f"Maximum: {max_bytes / (1024*1024):.1f} MB."
            )

        actual_size = os.path.getsize(output_path)
        print(f"  -> Downloaded: {actual_size / (1024*1024):.1f} MB")

        if expected_size > 0 and actual_size < expected_size * 0.99:
            print(f"  -> WARNING: File may be incomplete! Expected {expected_size}, got {actual_size}")
            return False

        if actual_size < 1000:
            print(f"  -> ERROR: Downloaded file too small ({actual_size} bytes)")
            return False

        print(f"  -> Download complete: {output_path}")
        return True
    except InputRejectedError:
        raise
    except Exception as e:
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except Exception:
            pass
        print(f"  -> ERROR: {e}")
        return False


def get_video_duration_seconds(video_path: str) -> float:
    """Best-effort duration via ffprobe. Returns 0 on failure."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return 0.0
        value = (result.stdout or "").strip()
        return float(value) if value else 0.0
    except Exception:
        return 0.0


def _extract_frames_raw(video_path: str, output_dir: str, fps: int, max_image_size: Optional[int] = None) -> int:
    os.makedirs(output_dir, exist_ok=True)

    vf_parts = [f"fps={fps}"]
    if max_image_size and max_image_size > 0:
        # Limit the larger image dimension to max_image_size while preserving aspect ratio (no upscaling).
        vf_parts.append(
            "scale="
            f"'if(gt(iw,ih),min(iw,{max_image_size}),-2)':"
            f"'if(gt(iw,ih),-2,min(ih,{max_image_size}))'"
        )

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", ",".join(vf_parts),
        "-q:v", "2",
        os.path.join(output_dir, "frame_%04d.jpg"),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  -> FFmpeg error: {result.stderr}")
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")

    return len(list(Path(output_dir).glob("*.jpg")))


def _select_frames(
    raw_dir: str,
    output_dir: str,
    target_frames: int,
    blur_threshold: Optional[float],
    dedupe_threshold: Optional[float],
) -> int:
    raw_frames = sorted(Path(raw_dir).glob("frame_*.jpg"))
    raw_count = len(raw_frames)
    print(f"  -> Raw frames: {raw_count}")

    if raw_count == 0:
        return 0

    # Validate thresholds
    blur_threshold = float(blur_threshold) if blur_threshold is not None else None
    dedupe_threshold = float(dedupe_threshold) if dedupe_threshold is not None else None

    def filter_frames(
        blur_thresh: Optional[float],
        dedupe_thresh: Optional[float],
    ) -> tuple[list[Path], int, int]:
        selected_local: list[Path] = []
        skipped_blur_local = 0
        skipped_dupe_local = 0
        prev_thumb_local: Optional[np.ndarray] = None

        for frame_path in raw_frames:
            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if blur_thresh is not None:
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                if sharpness < blur_thresh:
                    skipped_blur_local += 1
                    continue

            if dedupe_thresh is not None:
                thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
                if prev_thumb_local is not None:
                    diff = float(np.mean(np.abs(thumb.astype(np.float32) - prev_thumb_local.astype(np.float32))))
                    if diff < dedupe_thresh:
                        skipped_dupe_local += 1
                        continue
                prev_thumb_local = thumb

            selected_local.append(frame_path)

        return selected_local, skipped_blur_local, skipped_dupe_local

    selected, skipped_blur, skipped_dupe = filter_frames(blur_threshold, dedupe_threshold)

    # If filtering is too aggressive, relax constraints to ensure COLMAP gets enough images.
    # Heuristic: keep at least ~10% of target frames (capped by raw_count), but never less than 30.
    #
    # Rationale: For shaky/blurred videos, "high" presets can otherwise force the selector to
    # include many low-quality frames just to meet an overly high minimum, which hurts SfM.
    min_required = min(raw_count, max(30, int(max(1, target_frames) * 0.10)))

    if len(selected) < min_required:
        print(
            f"  -> WARNING: Filtering left only {len(selected)} frames (need ~{min_required}). "
            "Relaxing thresholds for robustness."
        )
        # First relax blur (keep dedupe).
        relaxed_blur = blur_threshold * 0.6 if blur_threshold is not None else None
        selected, skipped_blur, skipped_dupe = filter_frames(relaxed_blur, dedupe_threshold)

    if len(selected) < min_required:
        # Next disable blur filtering (keep dedupe).
        selected, skipped_blur, skipped_dupe = filter_frames(None, dedupe_threshold)

    if len(selected) < min_required:
        # Last resort: no filtering at all (but still downsample below).
        print("  -> WARNING: Frame filtering still too aggressive; falling back to unfiltered frames.")
        selected = raw_frames
        skipped_blur = 0
        skipped_dupe = 0

    if skipped_blur:
        print(f"  -> Removed blurry frames: {skipped_blur}")
    if skipped_dupe:
        print(f"  -> Removed near-duplicates: {skipped_dupe}")

    # Downsample to target_frames uniformly.
    target = max(1, int(target_frames))
    if len(selected) > target:
        step = len(selected) / target
        selected = [selected[int(i * step)] for i in range(target)]

    # Materialize selected frames into output_dir with contiguous names.
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for idx, src in enumerate(selected):
        dst = Path(output_dir) / f"frame_{idx + 1:04d}.jpg"
        try:
            # Prefer hardlinks so we can delete the raw frame directory safely.
            os.link(src, dst)
        except OSError:
            # Fallback to a full copy (e.g. on cross-device links).
            shutil.copy2(src, dst)

    print(f"  -> Selected frames: {len(selected)}")
    return len(selected)


def extract_frames(
    video_path: str,
    output_dir: str,
    target_frames: Optional[int] = None,
    max_extract_fps: int = 3,
    max_image_size: Optional[int] = None,
    blur_threshold: Optional[float] = None,
    dedupe_threshold: Optional[float] = None,
) -> int:
    """
    Extract frames with video-aware sampling + optional quality selection.

    - If target_frames is provided, compute an FPS based on video duration and oversample slightly.
    - Optionally filter blurry and near-duplicate frames.
    """
    duration = get_video_duration_seconds(video_path)
    fps = 3
    if target_frames and duration > 0:
        desired_fps = float(target_frames) / max(duration, 1.0)
        fps = int(math.ceil(desired_fps * 1.5))
        fps = max(2, min(int(max_extract_fps or 3), fps))
    else:
        fps = max(2, int(max_extract_fps or 3))

    if target_frames:
        print(f"[2/5] Extracting frames (target {target_frames}) at {fps} FPS")
    else:
        print(f"[2/5] Extracting frames at {fps} FPS")

    # If no selection is needed, extract directly into output_dir.
    selection_enabled = target_frames is not None or blur_threshold is not None or dedupe_threshold is not None
    if not selection_enabled:
        count = _extract_frames_raw(video_path, output_dir, fps=fps, max_image_size=max_image_size)
        print(f"  -> Extracted {count} frames")
        return count

    raw_dir = f"{output_dir}_raw"
    if os.path.isdir(raw_dir):
        shutil.rmtree(raw_dir)

    raw_count = _extract_frames_raw(video_path, raw_dir, fps=fps, max_image_size=max_image_size)
    if raw_count == 0:
        print("  -> ERROR: No frames extracted")
        return 0

    if target_frames is None:
        target_frames = min(raw_count, 300)

    selected_count = _select_frames(
        raw_dir=raw_dir,
        output_dir=output_dir,
        target_frames=target_frames,
        blur_threshold=blur_threshold,
        dedupe_threshold=dedupe_threshold,
    )

    try:
        shutil.rmtree(raw_dir)
    except Exception:
        pass

    return selected_count


def should_use_colmap_gpu() -> bool:
    value = os.environ.get("COLMAP_USE_GPU", "1").strip().lower()
    return value not in ("0", "false", "no")


def reset_colmap_workspace(database_path: str, sparse_dir: str) -> None:
    try:
        if os.path.exists(database_path):
            os.remove(database_path)
        if os.path.exists(sparse_dir):
            shutil.rmtree(sparse_dir)
        os.makedirs(sparse_dir, exist_ok=True)
    except Exception as cleanup_error:
        print(f"  -> WARNING: Failed to reset COLMAP workspace: {cleanup_error}")


def get_colmap_db_stats(database_path: str) -> Dict[str, int]:
    """Return basic COLMAP SQLite DB stats (best-effort)."""
    # Note: COLMAP stores keypoints/descriptors per-image as BLOBs; we report the total number of
    # extracted features by summing the per-image "rows" field (not the number of DB rows).
    stats: Dict[str, int] = {"images": 0, "keypoints": 0, "descriptors": 0}
    try:
        import sqlite3

        if not os.path.exists(database_path):
            return stats

        # Open read-only; avoid creating an empty DB file when COLMAP failed upstream.
        conn = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
        cur = conn.cursor()

        try:
            cur.execute("SELECT COUNT(*) FROM images")
            row = cur.fetchone()
            stats["images"] = int(row[0]) if row else 0
        except Exception:
            stats["images"] = 0

        # Total keypoints/descriptors across all images.
        for table_name in ("keypoints", "descriptors"):
            try:
                cur.execute(f"SELECT COALESCE(SUM(rows), 0) FROM {table_name}")
                row = cur.fetchone()
                stats[table_name] = int(row[0]) if row and row[0] is not None else 0
            except Exception:
                stats[table_name] = 0
        conn.close()
    except Exception as e:
        print(f"  -> WARNING: Could not read COLMAP database stats: {e}")
    return stats


def run_colmap(
    image_dir: str,
    workspace_dir: str,
    use_gpu: bool = True,
    matcher: str = "sequential",
    sequential_overlap: int = 12,
    loop_closure: bool = False,
    max_num_features: Optional[int] = None,
    max_image_size: Optional[int] = None,
    guided_matching: bool = True,
) -> bool:
    """Run COLMAP Structure-from-Motion pipeline."""
    print("[3/5] Running COLMAP SfM")

    database_path = os.path.join(workspace_dir, "database.db")
    sparse_dir = os.path.join(workspace_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    gpu_flag = "1" if use_gpu else "0"
    mode_label = "GPU" if use_gpu else "CPU"

    # Feature extraction
    print(f"  -> Extracting features ({mode_label})...")
    cmd = [
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", gpu_flag
    ]
    if max_num_features:
        cmd += ["--SiftExtraction.max_num_features", str(int(max_num_features))]
    if max_image_size:
        cmd += ["--SiftExtraction.max_image_size", str(int(max_image_size))]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        combined = "\n".join([result.stderr or "", result.stdout or ""]).strip()
        print(f"  -> Feature extraction failed: {combined}")
        if use_gpu:
            print("  -> GPU feature extraction failed, retrying with CPU...")
            reset_colmap_workspace(database_path, sparse_dir)
            return run_colmap(
                image_dir,
                workspace_dir,
                use_gpu=False,
                matcher=matcher,
                sequential_overlap=sequential_overlap,
                loop_closure=loop_closure,
                max_num_features=max_num_features,
                max_image_size=max_image_size,
                guided_matching=guided_matching,
            )
        return False

    # Sanity check: COLMAP must have created a database file.
    if not os.path.exists(database_path):
        combined = "\n".join([result.stderr or "", result.stdout or ""]).strip()
        print("  -> Feature extraction did not create a COLMAP database (database.db missing).")
        if combined:
            print(f"  -> COLMAP output: {combined}")
        if use_gpu:
            print("  -> Retrying feature extraction with CPU...")
            reset_colmap_workspace(database_path, sparse_dir)
            return run_colmap(
                image_dir,
                workspace_dir,
                use_gpu=False,
                matcher=matcher,
                sequential_overlap=sequential_overlap,
                loop_closure=loop_closure,
                max_num_features=max_num_features,
                max_image_size=max_image_size,
                guided_matching=guided_matching,
            )
        return False

    stats = get_colmap_db_stats(database_path)
    print(f"  -> Database stats: images={stats['images']}, keypoints={stats['keypoints']}, descriptors={stats['descriptors']}")
    if stats["images"] < 2:
        print("  -> Not enough images in database for matching (need at least 2).")
        return False
    if stats["keypoints"] == 0 or stats["descriptors"] == 0:
        print("  -> No keypoints/descriptors found after feature extraction. Check extracted frames quality.")
        return False

    # Feature matching
    matcher_key = (matcher or "sequential").strip().lower()
    if matcher_key not in ("exhaustive", "sequential"):
        matcher_key = "sequential"

    print(f"  -> Matching features ({mode_label}, {matcher_key})...")
    if matcher_key == "sequential":
        base_cmd = [
            "colmap", "sequential_matcher",
            "--database_path", database_path,
            "--SiftMatching.use_gpu", gpu_flag,
            "--SequentialMatching.overlap", str(int(sequential_overlap or 12)),
        ]

        cmd = list(base_cmd)
        if guided_matching:
            cmd += ["--SiftMatching.guided_matching", "1"]
        # NOTE: Some COLMAP builds default loop detection to "on" for sequential matching, which
        # requires a vocab tree file and can crash with visual_index.h:file.is_open(). We explicitly
        # disable it unless the caller opts in via loop_closure.
        cmd += ["--SequentialMatching.loop_detection", "1" if loop_closure else "0"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        combined_output = "\n".join([result.stderr or "", result.stdout or ""]).strip()
        combined_lower = combined_output.lower()

        if result.returncode != 0:
            if "unrecognized option" in combined_lower or "unknown option" in combined_lower or "unknown argument" in combined_lower:
                print("  -> loop_detection flag not supported by this COLMAP build, retrying without it...")
                retry_cmd = list(base_cmd)
                if guided_matching:
                    retry_cmd += ["--SiftMatching.guided_matching", "1"]
                result = subprocess.run(retry_cmd, capture_output=True, text=True)
                combined_output = "\n".join([result.stderr or "", result.stdout or ""]).strip()
                combined_lower = combined_output.lower()
            elif "visual_index.h" in combined_lower and "file.is_open" in combined_lower:
                # When loop detection is enabled (either explicitly or by default), COLMAP needs a vocab tree.
                # Retry with loop detection force-disabled.
                print("  -> Loop detection requires a vocab tree file; retrying with loop detection disabled...")
                retry_cmd = list(base_cmd)
                if guided_matching:
                    retry_cmd += ["--SiftMatching.guided_matching", "1"]
                retry_cmd += ["--SequentialMatching.loop_detection", "0"]
                result = subprocess.run(retry_cmd, capture_output=True, text=True)
                combined_output = "\n".join([result.stderr or "", result.stdout or ""]).strip()
                combined_lower = combined_output.lower()

        if result.returncode != 0 and guided_matching:
            if "cache.h" in combined_lower and "max_num_elems" in combined_lower:
                print("  -> Retrying sequential matcher without guided matching...")
                result = subprocess.run(list(base_cmd), capture_output=True, text=True)
                combined_output = "\n".join([result.stderr or "", result.stdout or ""]).strip()
                combined_lower = combined_output.lower()
    else:
        cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", database_path,
            "--SiftMatching.use_gpu", gpu_flag,
        ]
        if guided_matching:
            cmd += ["--SiftMatching.guided_matching", "1"]
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        combined_output = "\n".join([result.stderr or "", result.stdout or ""]).strip()
        print(f"  -> Matching failed: {combined_output}")

        # Some COLMAP builds crash inside sequential matcher with a cache assertion. Fall back to exhaustive matching.
        combined_lower = combined_output.lower()
        if matcher_key == "sequential" and ("cache.h" in combined_lower and "max_num_elems" in combined_lower):
            print("  -> Sequential matcher crashed (cache assertion). Retrying with exhaustive matcher...")
            fallback_cmd = [
                "colmap", "exhaustive_matcher",
                "--database_path", database_path,
                "--SiftMatching.use_gpu", gpu_flag,
            ]
            if guided_matching:
                fallback_cmd += ["--SiftMatching.guided_matching", "1"]
            fallback = subprocess.run(fallback_cmd, capture_output=True, text=True)
            if fallback.returncode == 0:
                print("  -> Exhaustive matcher succeeded.")
                result = fallback
            else:
                fallback_out = "\n".join([fallback.stderr or "", fallback.stdout or ""]).strip()
                print(f"  -> Exhaustive matcher also failed: {fallback_out}")

        # If matching still failed, try CPU fallback (only when GPU was used).
        if result.returncode != 0:
            if use_gpu:
                print("  -> GPU matching failed, retrying with CPU...")
                reset_colmap_workspace(database_path, sparse_dir)
                return run_colmap(
                    image_dir,
                    workspace_dir,
                    use_gpu=False,
                    matcher=matcher,
                    sequential_overlap=sequential_overlap,
                    loop_closure=loop_closure,
                    max_num_features=max_num_features,
                    max_image_size=max_image_size,
                    guided_matching=guided_matching,
                )
            return False

    # Sparse reconstruction (mapper)
    print("  -> Running mapper...")
    cmd = [
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--output_path", sparse_dir
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        combined = "\n".join([result.stderr or "", result.stdout or ""]).strip()
        print(f"  -> Mapper failed: {combined}")
        return False

    # Check if reconstruction was successful
    recon_dirs = list(Path(sparse_dir).iterdir())
    if not recon_dirs:
        print("  -> No reconstruction created")
        return False

    print(f"  -> COLMAP completed: {len(recon_dirs)} reconstruction(s)")
    return True


def ensure_opensplat_images(frames_dir: str, sparse_dir: str) -> None:
    """Ensure OpenSplat can find images under sparse_dir/images."""
    images_dir = os.path.join(sparse_dir, "images")
    existing = list(Path(images_dir).glob("*.jpg")) if os.path.isdir(images_dir) else []
    if existing:
        return

    os.makedirs(images_dir, exist_ok=True)
    frame_paths = list(Path(frames_dir).glob("*.jpg"))
    print(f"  -> Linking {len(frame_paths)} frames into {images_dir}")
    for frame_path in frame_paths:
        target_path = Path(images_dir) / frame_path.name
        if target_path.exists():
            continue
        try:
            os.link(frame_path, target_path)
        except OSError:
            shutil.copy2(frame_path, target_path)


def run_opensplat(frames_dir: str, colmap_dir: str, output_path: str, iterations: int = 7000) -> bool:
    """Run OpenSplat Gaussian Splatting training."""
    print(f"[4/5] Running OpenSplat ({iterations} iterations)")

    sparse_dir = os.path.join(colmap_dir, "sparse", "0")
    if not os.path.exists(sparse_dir):
        # Try without the "0" subdirectory
        sparse_dir = os.path.join(colmap_dir, "sparse")

    ensure_opensplat_images(frames_dir, sparse_dir)

    cmd = [
        "opensplat",
        sparse_dir,
        "-o", output_path,
        "-n", str(iterations),
        "--save-every", "-1"  # Only save final model
    ]

    # Stream output so RunPod logs show progress and we don't buffer huge stdout/stderr in memory.
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    output_lines = deque(maxlen=40)
    if process.stdout is not None:
        for line in process.stdout:
            line = (line or "").rstrip()
            if line:
                print(f"  [OpenSplat] {line}")
                output_lines.append(line)
    returncode = process.wait()

    if returncode != 0:
        tail = "\n".join(output_lines)
        print(f"  -> OpenSplat failed with exit code {returncode}")
        if tail:
            print(f"  -> OpenSplat output (tail):\n{tail}")
        return False

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  -> Model saved: {output_path} ({size_mb:.1f} MB)")
        return True
    else:
        print("  -> Model file not created")
        return False


def upload_to_s3(file_path: str, config: Dict[str, str], key: str) -> Optional[str]:
    """Upload file to S3/MinIO and return URL."""
    print("[5/5] Uploading to storage...")
    try:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=config["endpoint"],
            aws_access_key_id=config["access_key"],
            aws_secret_access_key=config["secret_key"],
            region_name=config.get("region", "us-east-1"),
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )

        extra_args = {}
        lower_path = file_path.lower()
        if lower_path.endswith(".ply"):
            extra_args["ContentType"] = "application/x-ply"
        elif lower_path.endswith(".splat") or lower_path.endswith(".spz"):
            extra_args["ContentType"] = "application/octet-stream"
        elif lower_path.endswith(".glb"):
            extra_args["ContentType"] = "model/gltf-binary"
        elif lower_path.endswith(".jpg") or lower_path.endswith(".jpeg"):
            extra_args["ContentType"] = "image/jpeg"

        if extra_args:
            client.upload_file(file_path, config["bucket"], key, ExtraArgs=extra_args)
        else:
            client.upload_file(file_path, config["bucket"], key)

        # Generate presigned URL (valid for 7 days)
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": config["bucket"], "Key": key},
            ExpiresIn=604800
        )
        print(f"  -> Uploaded: {key}")
        return url
    except Exception as e:
        print(f"  -> Upload error: {e}")
        return None


def get_video_extension(url: str) -> str:
    """Extract video extension from URL, defaulting to mp4."""
    # Parse URL to get path without query parameters
    from urllib.parse import urlparse, unquote
    parsed = urlparse(url)
    path = unquote(parsed.path)

    # Get extension from path
    ext = path.split('.')[-1].lower() if '.' in path else ''

    # Validate it's a known video extension
    valid_extensions = ['mp4', 'mov', 'avi', 'webm', 'mkv', 'm4v']
    if ext in valid_extensions:
        return ext

    # Default to mp4
    return 'mp4'


def slugify(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "scan"


def ensure_s3_config(upload_config: Optional[Dict[str, str]]) -> Optional[str]:
    if not upload_config:
        return "S3 upload is not configured. Set S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET."

    required = ["endpoint", "access_key", "secret_key", "bucket"]
    missing = [k for k in required if not upload_config.get(k)]
    if missing:
        return f"Missing S3 config: {', '.join(missing)}. Required: S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET."

    return None


def process_video_to_3d(
    video_url: Optional[str] = None,
    video_key: Optional[str] = None,
    title: str = "scan",
    output_format: str = "ply",
    iterations: Optional[int] = None,
    quality_preset: str = "balanced",
    target_frames: Optional[int] = None,
    max_image_size: Optional[int] = None,
    max_extract_fps: Optional[int] = None,
    blur_threshold: Optional[float] = None,
    dedupe_threshold: Optional[float] = None,
    colmap_matcher: Optional[str] = None,
    sequential_overlap: Optional[int] = None,
    loop_closure: Optional[bool] = None,
    max_num_features: Optional[int] = None,
    guided_matching: Optional[bool] = None,
    upload_config: Optional[Dict[str, str]] = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main processing pipeline.

    Args:
        video_url: URL to source video
        title: Scan title for naming
        output_format: 'ply' or 'splat'
        iterations: Training iterations
        upload_config: S3/MinIO config for uploading results

    Returns:
        Dict with model_url, preview_url, status
    """
    config_error = ensure_s3_config(upload_config)
    if config_error:
        return {"error": config_error, "status": "failed"}

    work_dir = tempfile.mkdtemp(prefix="gs_")

    try:
        params = get_quality_params(quality_preset)
        if target_frames is not None:
            params["target_frames"] = int(target_frames)
        if max_image_size is not None:
            params["max_image_size"] = int(max_image_size)
        if max_extract_fps is not None:
            params["max_extract_fps"] = int(max_extract_fps)
        if blur_threshold is not None:
            params["blur_threshold"] = float(blur_threshold)
        if dedupe_threshold is not None:
            params["dedupe_threshold"] = float(dedupe_threshold)
        if colmap_matcher is not None:
            params["colmap_matcher"] = str(colmap_matcher)
        if sequential_overlap is not None:
            params["sequential_overlap"] = int(sequential_overlap)
        if loop_closure is not None:
            params["loop_closure"] = bool(loop_closure)
        if max_num_features is not None:
            params["max_num_features"] = int(max_num_features)
        if guided_matching is not None:
            params["guided_matching"] = bool(guided_matching)
        if iterations is not None:
            params["iterations"] = int(iterations)

        iterations_final = int(params.get("iterations") or 7000)

        # Detect video extension from URL to preserve original format
        source_for_ext = video_url or video_key or "input.mp4"
        video_ext = get_video_extension(source_for_ext)
        video_path = os.path.join(work_dir, f"input.{video_ext}")
        print(f"  -> Detected video format: {video_ext}")
        print(
            "  -> Quality preset: "
            f"{(quality_preset or 'balanced').strip().lower()} "
            f"(frames={params.get('target_frames')}, "
            f"matcher={params.get('colmap_matcher')}, "
            f"iters={iterations_final})"
        )
        frames_dir = os.path.join(work_dir, "frames")
        colmap_dir = os.path.join(work_dir, "colmap")
        output_path = os.path.join(work_dir, f"model.{output_format}")

        os.makedirs(colmap_dir, exist_ok=True)

        # Step 1: Download video
        downloaded = False
        try:
            if video_key:
                downloaded = download_video_from_s3(video_key, video_path, upload_config) if upload_config else False
                if not downloaded and video_url:
                    print("  -> S3 download failed, trying presigned URL...")
                    downloaded = download_video(video_url, video_path)
            elif video_url:
                inferred_key = extract_s3_key_from_url(video_url, upload_config["bucket"]) if upload_config else None
                if inferred_key and upload_config:
                    downloaded = download_video_from_s3(inferred_key, video_path, upload_config)
                    if not downloaded:
                        print("  -> S3 download failed, trying presigned URL...")
                if not downloaded:
                    downloaded = download_video(video_url, video_path)
        except InputRejectedError as e:
            return {"error": str(e), "status": "failed"}

        if not downloaded:
            return {"error": "Failed to download video", "status": "failed"}

        # Fail fast on overly long videos (saves expensive SfM/training time).
        max_seconds = get_max_input_video_seconds()
        if max_seconds > 0:
            duration = get_video_duration_seconds(video_path)
            if duration > 0:
                print(f"  -> Video duration: {duration:.1f}s")
                if duration > max_seconds:
                    return {
                        "error": f"Video too long ({int(round(duration))}s). Maximum: {max_seconds}s.",
                        "status": "failed",
                    }

        # Step 2: Extract frames
        frame_count = extract_frames(
            video_path,
            frames_dir,
            target_frames=int(params.get("target_frames") or 0) or None,
            max_extract_fps=int(params.get("max_extract_fps") or 3),
            max_image_size=int(params.get("max_image_size") or 0) or None,
            blur_threshold=params.get("blur_threshold"),
            dedupe_threshold=params.get("dedupe_threshold"),
        )
        if frame_count < 10:
            return {"error": f"Not enough frames ({frame_count}). Need at least 10.", "status": "failed"}

        # If the caller did not explicitly override iterations, adapt training length to the actual
        # number of selected frames. This avoids long training runs (and RunPod execution timeouts)
        # when the video is very blurry and only few frames survive quality filtering.
        if iterations is None:
            adjusted_iters = iterations_final
            if frame_count < 60:
                adjusted_iters = min(adjusted_iters, 5000)
            elif frame_count < 120:
                adjusted_iters = min(adjusted_iters, 7000)

            if adjusted_iters != iterations_final:
                print(
                    "  -> Adjusting OpenSplat iterations "
                    f"from {iterations_final} to {adjusted_iters} "
                    f"(selected_frames={frame_count}, preset={(quality_preset or 'balanced').strip().lower()})"
                )
                iterations_final = adjusted_iters

        # Step 3: Run COLMAP
        os.environ.setdefault("DISPLAY", ":99")
        if not run_colmap(
            frames_dir,
            colmap_dir,
            use_gpu=should_use_colmap_gpu(),
            matcher=str(params.get("colmap_matcher") or "sequential"),
            sequential_overlap=int(params.get("sequential_overlap") or 12),
            loop_closure=bool(params.get("loop_closure")),
            max_num_features=int(params.get("max_num_features") or 0) or None,
            max_image_size=int(params.get("max_image_size") or 0) or None,
            guided_matching=bool(params.get("guided_matching", True)),
        ):
            return {"error": "COLMAP reconstruction failed", "status": "failed"}

        # Step 4: Run OpenSplat
        if not run_opensplat(frames_dir, colmap_dir, output_path, iterations_final):
            return {"error": "OpenSplat training failed", "status": "failed"}

        # Step 5: Upload results (optional)
        if not os.path.exists(output_path):
            return {"error": "Model file not created", "status": "failed"}

        safe_title = slugify(title)
        stable_id = job_id or str(uuid.uuid4())
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        key_prefix = f"3d-provider/gaussian-splatting/{safe_title}/{stable_id}-{ts}"

        model_key = f"{key_prefix}/model.{output_format}"
        model_url = upload_to_s3(output_path, upload_config, model_key) if upload_config else None
        if not model_url:
            return {"error": "Failed to upload model to S3/MinIO", "status": "failed"}

        preview_url = None
        try:
            from PIL import Image

            frames = sorted(Path(frames_dir).glob("*.jpg"))
            if frames:
                preview_path = os.path.join(work_dir, "preview.jpg")
                img = Image.open(str(frames[len(frames) // 2]))
                img = img.convert("RGB")
                img.thumbnail((1280, 1280))
                img.save(preview_path, format="JPEG", quality=85, optimize=True)

                preview_key = f"{key_prefix}/preview.jpg"
                preview_url = upload_to_s3(preview_path, upload_config, preview_key) if upload_config else None
        except Exception as preview_error:
            print(f"  -> Preview generation/upload failed: {preview_error}")

        return {
            "status": "completed",
            "model_url": model_url,
            "preview_url": preview_url,
            "format": output_format,
            "model_key": model_key,
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "status": "failed",
            "traceback": traceback.format_exc(),
        }
    finally:
        # Cleanup
        try:
            shutil.rmtree(work_dir)
        except:
            pass


if __name__ == "__main__":
    # Test locally
    import sys
    if len(sys.argv) > 1:
        result = process_video_to_3d(sys.argv[1])
        import json
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python process_video.py <video_url>")
