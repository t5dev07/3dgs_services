"""Depth Anything V2 → per-frame 16-bit PNG depth maps for Nerfstudio/dn-splatter.

Strategy:
  - Run DA2-Small ONNX on every Nth frame (configurable).
  - Scale-align each frame's relative depth to COLMAP metric scale using
    2D-3D correspondences from the COLMAP sparse model.
  - Write uint16 PNGs (pixel value = depth / unit_scale_factor, clipped to 65535).
  - dn-splatter reads these via transforms.json `depth_file_path` and applies
    a depth-supervision loss during training.

Non-fatal: missing model / failed scale estimation / any error → log warning and skip.
"""
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from settings import settings

from .context import PipelineContext, ProgressCallback
from .depth_utils import (
    Camera,
    Image as _Image,
    colmap_model_converter,
    find_best_sparse_dir,
    parse_cameras_txt,
    parse_images_txt,
    parse_points3d_txt,
)

log = logging.getLogger(__name__)

DA2_INPUT_SIZE = 518              # DA2-Small ViT patch size multiple
MIN_DEPTH_NORMALIZED = 0.05       # skip near-zero DA2 samples (noisy)


# ---------------------------------------------------------------------------
# DA2 inference
# ---------------------------------------------------------------------------

def _load_session(model_path: Path):
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        log.warning("onnxruntime not installed, skipping depth maps")
        return None

    if not model_path.exists():
        log.warning("DA2 model not found at %s, skipping depth maps", model_path)
        return None

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(str(model_path), providers=providers)
        log.info("DA2-Small loaded — provider: %s", session.get_providers()[0])
        return session
    except Exception as exc:
        log.warning("Failed to load DA2 model: %s", exc)
        return None


def _infer_depth(session, img_bgr: np.ndarray) -> np.ndarray:
    """Return DA2 relative-depth (H, W) normalized to [0, 1]."""
    h, w = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr, (DA2_INPUT_SIZE, DA2_INPUT_SIZE))
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
    blob = np.transpose(rgb, (2, 0, 1))[None]

    out = session.run(None, {session.get_inputs()[0].name: blob})[0]
    depth = out[0] if out.ndim == 3 else out[0, 0]

    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max - d_min < 1e-6:
        return np.zeros((h, w), dtype=np.float32)
    depth_norm = (depth - d_min) / (d_max - d_min)
    return cv2.resize(depth_norm.astype(np.float32), (w, h))


# ---------------------------------------------------------------------------
# Scale alignment
# ---------------------------------------------------------------------------

def _estimate_scale_meters(
    depth_norm: np.ndarray,
    img_data: _Image,
    camera: Camera,
    existing_pts: dict[int, np.ndarray],
    min_n: int,
) -> Optional[float]:
    """Return metric scale (meters) s.t. depth_meters = depth_norm * scale.

    Uses the median of (COLMAP_depth / DA2_depth) over matched 2D-3D points.
    """
    R = img_data.rotation_matrix()
    t = img_data.translation()
    h, w = depth_norm.shape
    ratios = []

    for u, v, pt_id in img_data.observations:
        if pt_id < 0 or pt_id not in existing_pts:
            continue
        p_cam = R @ existing_pts[pt_id] + t
        colmap_depth = float(p_cam[2])
        if colmap_depth <= 0:
            continue
        ui = int(np.clip(round(u * w / camera.width), 0, w - 1))
        vi = int(np.clip(round(v * h / camera.height), 0, h - 1))
        da2_val = float(depth_norm[vi, ui])
        if da2_val < MIN_DEPTH_NORMALIZED:
            continue
        ratios.append(colmap_depth / da2_val)

    if len(ratios) < min_n:
        return None
    return float(np.median(ratios))


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------

def run(ctx: PipelineContext, params: dict, on_progress: ProgressCallback) -> None:
    cfg = settings.pipeline.depth
    if not cfg.enabled:
        log.info("[%s] depth maps disabled via settings", ctx.job_id)
        return

    session = _load_session(Path(cfg.model_path))
    if session is None:
        return

    sparse_dir = find_best_sparse_dir(ctx.colmap_dir)
    if sparse_dir is None:
        log.warning("[%s] No COLMAP sparse dir, skipping depth maps", ctx.job_id)
        return
    log.info("[%s] Using sparse reconstruction: %s", ctx.job_id, sparse_dir)

    on_progress("depth", 40, "Generating metric depth maps...")

    # Convert sparse BIN → TXT so we can parse correspondences
    txt_dir = ctx.colmap_dir / "sparse_txt"
    if not colmap_model_converter(sparse_dir, txt_dir, "TXT", ctx.log_file):
        log.warning("[%s] COLMAP model_converter failed, skipping depth maps", ctx.job_id)
        return

    cameras = parse_cameras_txt(txt_dir / "cameras.txt")
    images = parse_images_txt(txt_dir / "images.txt")
    existing_pts = parse_points3d_txt(txt_dir / "points3D.txt")
    if not images or not cameras or not existing_pts:
        log.warning("[%s] Empty COLMAP model, skipping depth maps", ctx.job_id)
        return

    img_by_name = {img.name: img for img in images}
    ctx.depths_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_scale = 0
    for i, img_data in enumerate(images):
        if cfg.skip_frames > 1 and (i % cfg.skip_frames != 0):
            continue

        frame_path = ctx.frames_dir / img_data.name
        if not frame_path.exists():
            continue
        camera = cameras.get(img_data.camera_id)
        if camera is None:
            continue

        img_bgr = cv2.imread(str(frame_path))
        if img_bgr is None:
            continue

        depth_norm = _infer_depth(session, img_bgr)
        scale_m = _estimate_scale_meters(depth_norm, img_data, camera, existing_pts, cfg.min_correspondences)
        if scale_m is None:
            skipped_scale += 1
            continue

        depth_mm = depth_norm * scale_m / cfg.unit_scale_factor
        depth_uint16 = np.clip(depth_mm, 0, 65535).astype(np.uint16)

        out_path = ctx.depths_dir / f"{Path(img_data.name).stem}.png"
        cv2.imwrite(str(out_path), depth_uint16)
        written += 1

    on_progress("depth", 42, f"Depth maps: {written} written, {skipped_scale} skipped (scale)")
    log.info(
        "[%s] Depth maps: %d written, %d skipped (no scale alignment)",
        ctx.job_id, written, skipped_scale,
    )
