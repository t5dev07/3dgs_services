"""Depth Anything V2 depth maps → augment COLMAP sparse point cloud.

Strategy:
  - Run DA2-Small ONNX on a sample of extracted frames to get relative depth maps.
  - Scale-align each frame's depth to COLMAP metric scale using 2D-3D correspondences
    already present in the COLMAP model (images.txt point observations).
  - Back-project depth pixels to 3D world coordinates using COLMAP camera poses.
  - Augment the COLMAP sparse reconstruction (points3D.txt) with these new points
    (empty track — OpenSplat only reads xyz+rgb from points3D, discards track data).
  - Feed the denser point cloud to OpenSplat for better Gaussian initialization.

Non-fatal: if DA2 model is missing, scale alignment fails, or any step errors,
the function logs a warning and returns without modifying the COLMAP output.
"""
import logging
import math
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .context import PipelineContext, ProgressCallback

log = logging.getLogger(__name__)

MODEL_PATH = Path("/app/models/depth_anything_v2_vits.onnx")
DA2_INPUT_SIZE = 518        # DA2-Small ViT patch size multiple
SKIP_FRAMES = 4             # process every Nth frame (speed/quality balance)
POINTS_PER_FRAME = 1000     # max new 3D points per frame
MIN_DEPTH_NORMALIZED = 0.05 # ignore very-near depth samples (likely noise)
MIN_SCALE_SAMPLES = 5       # minimum COLMAP correspondences needed for scale alignment


# ---------------------------------------------------------------------------
# DA2 ONNX inference
# ---------------------------------------------------------------------------

def _load_da2_session():
    """Load DA2-Small ONNX session. Returns None if not available."""
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        log.warning("onnxruntime not installed, skipping depth augmentation")
        return None

    if not MODEL_PATH.exists():
        log.warning("DA2 model not found at %s, skipping depth augmentation", MODEL_PATH)
        return None

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(str(MODEL_PATH), providers=providers)
        active = session.get_providers()[0]
        log.info("DA2-Small loaded — provider: %s", active)
        return session
    except Exception as exc:
        log.warning("Failed to load DA2 model: %s", exc)
        return None


def _infer_depth(session, img_bgr: np.ndarray) -> np.ndarray:
    """Run DA2 inference. Returns normalized depth map (H, W) in [0, 1]."""
    h, w = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr, (DA2_INPUT_SIZE, DA2_INPUT_SIZE))
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
    blob = np.transpose(rgb, (2, 0, 1))[None]  # (1,3,518,518)

    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: blob})[0]  # (1,H,W) or (1,1,H,W)
    depth = out[0] if out.ndim == 3 else out[0, 0]  # → (H, W)

    # Normalize to [0, 1]
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max - d_min < 1e-6:
        return np.zeros((h, w), dtype=np.float32)
    depth_norm = (depth - d_min) / (d_max - d_min)

    return cv2.resize(depth_norm.astype(np.float32), (w, h))


# ---------------------------------------------------------------------------
# COLMAP text-format parsers
# ---------------------------------------------------------------------------

def _find_sparse_dir(colmap_dir: Path) -> Optional[Path]:
    """Return path to COLMAP sparse reconstruction directory."""
    for candidate in [colmap_dir / "sparse" / "0", colmap_dir / "sparse"]:
        if candidate.exists():
            return candidate
    return None


def _colmap_model_converter(src: Path, dst: Path, output_type: str, log_file: Path) -> bool:
    dst.mkdir(parents=True, exist_ok=True)
    cmd = [
        "colmap", "model_converter",
        "--input_path", str(src),
        "--output_path", str(dst),
        "--output_type", output_type,
    ]
    with open(log_file, "a") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    return result.returncode == 0


class _Camera:
    __slots__ = ("model", "width", "height", "fx", "fy", "cx", "cy")

    def __init__(self, model, width, height, params):
        self.model = model
        self.width = width
        self.height = height
        # Parse intrinsics per model type
        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
            self.fx = self.fy = float(params[0])
            self.cx = float(params[1]); self.cy = float(params[2])
        elif model in ("PINHOLE", "OPENCV", "FULL_OPENCV", "RADIAL"):
            self.fx = float(params[0]); self.fy = float(params[1])
            self.cx = float(params[2]); self.cy = float(params[3])
        else:
            # Fallback: assume first param is focal length
            self.fx = self.fy = float(params[0])
            self.cx = width / 2.0; self.cy = height / 2.0


def _parse_cameras_txt(path: Path) -> dict[int, _Camera]:
    cameras: dict[int, _Camera] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = parts[4:]
            cameras[cam_id] = _Camera(model, width, height, params)
    return cameras


class _Image:
    __slots__ = ("image_id", "qvec", "tvec", "camera_id", "name", "observations")

    def __init__(self, image_id, qvec, tvec, camera_id, name, observations):
        self.image_id = image_id
        self.qvec = qvec          # (qw,qx,qy,qz)
        self.tvec = tvec          # (tx,ty,tz)
        self.camera_id = camera_id
        self.name = name
        self.observations = observations  # list of (u, v, point3d_id)

    def rotation_matrix(self) -> np.ndarray:
        """Quaternion (qw,qx,qy,qz) → 3×3 rotation matrix R (world→camera)."""
        qw, qx, qy, qz = self.qvec
        return np.array([
            [1-2*(qy*qy+qz*qz),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
            [  2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qw*qx)],
            [  2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)],
        ], dtype=np.float64)

    def translation(self) -> np.ndarray:
        return np.array(self.tvec, dtype=np.float64)


def _parse_images_txt(path: Path) -> list[_Image]:
    images = []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        # Line 1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        parts = lines[i].split()
        image_id = int(parts[0])
        qvec = tuple(float(x) for x in parts[1:5])
        tvec = tuple(float(x) for x in parts[5:8])
        camera_id = int(parts[8])
        name = parts[9]
        # Line 2: POINTS2D[] as (X Y POINT3D_ID) ...
        obs = []
        if i + 1 < len(lines):
            obs_parts = lines[i + 1].split()
            for j in range(0, len(obs_parts) - 2, 3):
                u = float(obs_parts[j])
                v = float(obs_parts[j + 1])
                pt_id = int(obs_parts[j + 2])
                obs.append((u, v, pt_id))
        images.append(_Image(image_id, qvec, tvec, camera_id, name, obs))
        i += 2
    return images


def _parse_points3d_txt(path: Path) -> dict[int, np.ndarray]:
    """Returns {point_id: xyz (float64 array)}."""
    pts: dict[int, np.ndarray] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pt_id = int(parts[0])
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            pts[pt_id] = xyz
    return pts


def _write_extra_points3d_txt(path: Path, new_points: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Append depth-derived points to existing points3D.txt (empty track)."""
    existing_lines = path.read_text() if path.exists() else "# 3D point list\n"
    # Determine max existing point_id to avoid collisions
    max_id = 0
    for line in existing_lines.splitlines():
        if line and not line.startswith("#"):
            try:
                max_id = max(max_id, int(line.split()[0]))
            except (ValueError, IndexError):
                pass

    with open(path, "a") as f:
        for i, (xyz, rgb) in enumerate(new_points):
            pt_id = max_id + i + 1
            r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
            # Format: POINT3D_ID X Y Z R G B ERROR (no track = empty)
            f.write(f"{pt_id} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} {r} {g} {b} 0.0\n")


# ---------------------------------------------------------------------------
# Scale alignment + backprojection
# ---------------------------------------------------------------------------

def _estimate_depth_scale(
    depth_norm: np.ndarray,
    img_data: _Image,
    camera: _Camera,
    existing_pts: dict[int, np.ndarray],
) -> Optional[float]:
    """Compute scale factor: COLMAP_depth / DA2_relative_depth at matched 2D points."""
    R = img_data.rotation_matrix()
    t = img_data.translation()
    h, w = depth_norm.shape
    ratios = []

    for u, v, pt_id in img_data.observations:
        if pt_id < 0 or pt_id not in existing_pts:
            continue
        # World → camera space
        p_cam = R @ existing_pts[pt_id] + t
        colmap_depth = float(p_cam[2])
        if colmap_depth <= 0:
            continue

        # Sample DA2 depth at pixel (u, v) — clamp to valid range
        ui = int(np.clip(round(u * w / camera.width), 0, w - 1))
        vi = int(np.clip(round(v * h / camera.height), 0, h - 1))
        da2_val = float(depth_norm[vi, ui])
        if da2_val < MIN_DEPTH_NORMALIZED:
            continue

        ratios.append(colmap_depth / da2_val)

    if len(ratios) < MIN_SCALE_SAMPLES:
        return None
    return float(np.median(ratios))


def _backproject(
    depth_metric: np.ndarray,
    img_bgr: np.ndarray,
    camera: _Camera,
    R: np.ndarray,
    t: np.ndarray,
    n_points: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Back-project depth pixels → 3D world points (xyz, rgb)."""
    h, w = depth_metric.shape
    valid_mask = depth_metric > 0.1  # skip near-zero depth
    ys, xs = np.where(valid_mask)
    if len(ys) == 0:
        return []

    n_sample = min(n_points, len(ys))
    idx = np.random.choice(len(ys), n_sample, replace=False)
    ys, xs = ys[idx], xs[idx]
    depths = depth_metric[ys, xs]

    # Pixel → camera space
    X = (xs - camera.cx) * depths / camera.fx
    Y = (ys - camera.cy) * depths / camera.fy
    Z = depths

    # Camera → world: P_world = R^T @ (P_cam − t)
    P_cam = np.stack([X, Y, Z], axis=1)          # (N, 3)
    P_world = (R.T @ (P_cam - t).T).T             # (N, 3)

    # Sample colors (BGR → RGB for points3D.txt)
    colors = img_bgr[ys, xs][:, ::-1].astype(np.uint8)

    return [(P_world[i], colors[i]) for i in range(n_sample)]


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------

def run(ctx: PipelineContext, params: dict, on_progress: ProgressCallback) -> None:
    """Augment COLMAP sparse point cloud using DA2 monocular depth maps.

    Non-fatal: any failure causes an early return while pipeline continues
    using the original (unaugmented) COLMAP reconstruction.
    """
    session = _load_da2_session()
    if session is None:
        return

    sparse_dir = _find_sparse_dir(ctx.colmap_dir)
    if sparse_dir is None:
        log.warning("[%s] No COLMAP sparse dir found, skipping depth aug", ctx.job_id)
        return

    on_progress("depth", 41, "Generating depth maps for point cloud densification...")

    try:
        _run_depth_augmentation(ctx, session, sparse_dir, on_progress)
    except Exception as exc:
        log.warning("[%s] Depth augmentation failed: %s — using original COLMAP", ctx.job_id, exc)


def _run_depth_augmentation(
    ctx: PipelineContext,
    session,
    sparse_dir: Path,
    on_progress: ProgressCallback,
) -> None:
    txt_dir = ctx.colmap_dir / "sparse_txt"
    aug_txt_dir = ctx.colmap_dir / "sparse_aug_txt"
    aug_bin_dir = ctx.colmap_dir / "sparse_aug"

    # 1. Convert binary → text
    if not _colmap_model_converter(sparse_dir, txt_dir, "TXT", ctx.log_file):
        log.warning("[%s] COLMAP model_converter (→TXT) failed", ctx.job_id)
        return

    # 2. Parse COLMAP model
    cameras = _parse_cameras_txt(txt_dir / "cameras.txt")
    images = _parse_images_txt(txt_dir / "images.txt")
    existing_pts = _parse_points3d_txt(txt_dir / "points3D.txt")

    if not images or not cameras or not existing_pts:
        log.warning("[%s] Empty COLMAP model, skipping depth aug", ctx.job_id)
        return

    # 3. Sample frames + run depth + scale-align + backproject
    frames = sorted(ctx.frames_dir.glob("frame_*.jpg"))
    # Build name→_Image lookup (COLMAP name is just the filename)
    img_by_name = {img.name: img for img in images}

    new_points: list[tuple[np.ndarray, np.ndarray]] = []
    processed = 0

    for i, frame_path in enumerate(frames):
        if i % SKIP_FRAMES != 0:
            continue

        img_data = img_by_name.get(frame_path.name)
        if img_data is None:
            continue
        if img_data.camera_id not in cameras:
            continue

        img_bgr = cv2.imread(str(frame_path))
        if img_bgr is None:
            continue

        depth_norm = _infer_depth(session, img_bgr)

        camera = cameras[img_data.camera_id]
        scale = _estimate_depth_scale(depth_norm, img_data, camera, existing_pts)
        if scale is None:
            log.debug("[%s] Depth scale align failed for %s", ctx.job_id, frame_path.name)
            continue

        depth_metric = depth_norm * scale
        R = img_data.rotation_matrix()
        t = img_data.translation()
        pts = _backproject(depth_metric, img_bgr, camera, R, t, POINTS_PER_FRAME)
        new_points.extend(pts)
        processed += 1

    if not new_points:
        log.warning("[%s] No depth-derived points generated, skipping aug", ctx.job_id)
        return

    log.info(
        "[%s] Depth aug: processed %d frames, generated %d new 3D points",
        ctx.job_id, processed, len(new_points),
    )

    # 4. Write augmented points3D.txt
    aug_txt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(txt_dir / "cameras.txt", aug_txt_dir / "cameras.txt")
    shutil.copy2(txt_dir / "images.txt", aug_txt_dir / "images.txt")
    shutil.copy2(txt_dir / "points3D.txt", aug_txt_dir / "points3D.txt")
    _write_extra_points3d_txt(aug_txt_dir / "points3D.txt", new_points)

    # 5. Convert augmented text → binary
    if not _colmap_model_converter(aug_txt_dir, aug_bin_dir, "BIN", ctx.log_file):
        log.warning("[%s] COLMAP model_converter (→BIN augmented) failed", ctx.job_id)
        return

    on_progress("depth", 45, f"Point cloud densified: +{len(new_points)} points from depth")
    log.info("[%s] Depth augmentation complete → %s", ctx.job_id, aug_bin_dir)
