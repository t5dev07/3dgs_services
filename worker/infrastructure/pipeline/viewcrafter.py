"""ViewCrafter — synthetic novel-view augmentation for sparse regions.

Flow:
  1. Load transforms.json (already has COLMAP poses from transforms.py step).
  2. Detect "sparse" keyframes = frames with < min_neighbors within sparse_radius_m.
  3. For each sparse keyframe: call ViewCrafter to render an orbit clip around
     the reference image, with trajectory specified as spherical deltas.
  4. Compose each synthetic frame's pose into COLMAP world = ref_c2w @ relative_c2w.
  5. Write synthetic JPEGs into ns_dir/images/, append entries to transforms.json
     with a shared soft-value mask to down-weight their training loss.

Non-fatal: disabled by default (settings.pipeline.viewcrafter.enabled = false).
When enabled, any failure (model load, inference, pose compose) logs a warning
and leaves transforms.json untouched.

Prerequisites for real invocation:
  - ViewCrafter repo cloned to /app/vendor/viewcrafter/ (not pip-installable).
  - ViewCrafter + DUSt3R weights in settings.pipeline.viewcrafter.model_dir.
  - pytorch3d installed (CUDA build matching base image torch).
"""
from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from settings import settings

from .context import PipelineContext, ProgressCallback

log = logging.getLogger(__name__)

VIEWCRAFTER_REPO = Path("/app/vendor/viewcrafter")


# ---------------------------------------------------------------------------
# Sparse keyframe detection
# ---------------------------------------------------------------------------

def _camera_center(transform_matrix: list[list[float]]) -> np.ndarray:
    """Extract camera center from 4×4 camera-to-world matrix (Nerfstudio convention)."""
    return np.asarray(transform_matrix, dtype=np.float64)[:3, 3]


def _find_sparse_keyframes(frames: list[dict], radius_m: float, min_neighbors: int) -> list[dict]:
    """Return frames with fewer than `min_neighbors` other frames within `radius_m`."""
    centers = np.stack([_camera_center(f["transform_matrix"]) for f in frames])  # (N, 3)
    sparse = []
    for i, f in enumerate(frames):
        dists = np.linalg.norm(centers - centers[i], axis=1)
        neighbors = int(np.sum((dists > 0) & (dists < radius_m)))
        if neighbors < min_neighbors:
            sparse.append(f)
    return sparse


# ---------------------------------------------------------------------------
# Pose composition
# ---------------------------------------------------------------------------

def _orbit_trajectory(n_frames: int, orbit_deg: float) -> list[np.ndarray]:
    """Yaw-orbit: return list of 4×4 relative c2w matrices rotating around Y axis.

    Frame 0 is identity. Frame i rotates by i/(n-1) * orbit_deg.
    """
    mats = []
    for i in range(n_frames):
        ang = math.radians(orbit_deg * i / max(1, n_frames - 1))
        c, s = math.cos(ang), math.sin(ang)
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        mats.append(m)
    return mats


def _compose(ref_c2w: list[list[float]], rel_c2w: np.ndarray) -> np.ndarray:
    return np.asarray(ref_c2w, dtype=np.float64) @ rel_c2w


# ---------------------------------------------------------------------------
# ViewCrafter invocation — runs vendored inference.py in single_view_target mode
# ---------------------------------------------------------------------------

def _extract_frames_from_mp4(mp4_path: Path, out_dir: Path) -> list[Path]:
    """Extract frames from a generated MP4 using ffmpeg. Returns list of PNG paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%04d.png"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4_path), str(pattern)]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("frame_*.png"))


def _call_viewcrafter(
    ref_image: Path,
    out_dir: Path,
    orbit_deg: float,
    n_frames: int,
    variant: str,
    model_dir: Path,
    dust3r_ckpt: Path,
    height: int = 320,
    width: int = 512,
    ddim_steps: int = 50,
) -> list[Path]:
    """Invoke ViewCrafter inference.py with single_view_target mode.

    Raises RuntimeError if repo or weights missing. Output trajectory goes
    from reference (d_phi=0) to d_phi=orbit_deg over n_frames frames.
    """
    if not VIEWCRAFTER_REPO.exists():
        raise RuntimeError(
            f"ViewCrafter repo not found at {VIEWCRAFTER_REPO}. "
            "Clone https://github.com/Drexubery/ViewCrafter into worker/vendor_viewcrafter/ "
            "and rebuild the image."
        )

    ckpt = model_dir / f"{variant}.ckpt"
    if not ckpt.exists():
        raise RuntimeError(
            f"ViewCrafter weights not found at {ckpt}. "
            "Run: VIEWCRAFTER_DOWNLOAD=1 python scripts/download_models.py"
        )
    if not dust3r_ckpt.exists():
        raise RuntimeError(f"DUSt3R weights not found at {dust3r_ckpt}")

    # 1024 variants use inference_pvd_1024.yaml; 512 variants use 512.yaml
    config_name = "inference_pvd_512.yaml" if width <= 512 else "inference_pvd_1024.yaml"
    config_path = VIEWCRAFTER_REPO / "configs" / config_name
    if not config_path.exists():
        # Fallback: 1024 config works at lower res too (just slower/memory-heavy)
        config_path = VIEWCRAFTER_REPO / "configs" / "inference_pvd_1024.yaml"

    out_dir.mkdir(parents=True, exist_ok=True)
    exp_name = ref_image.stem

    cmd = [
        sys.executable,
        str(VIEWCRAFTER_REPO / "inference.py"),
        "--image_dir", str(ref_image),
        "--out_dir", str(out_dir),
        "--exp_name", exp_name,
        "--mode", "single_view_target",
        "--d_phi", str(int(orbit_deg)),
        "--d_theta", "0",
        "--d_r", "0",
        "--d_x", "0",
        "--d_y", "0",
        "--video_length", str(n_frames),
        "--height", str(height),
        "--width", str(width),
        "--ddim_steps", str(ddim_steps),
        "--ckpt_path", str(ckpt),
        "--model_path", str(dust3r_ckpt),
        "--config", str(config_path),
        "--seed", "42",
        "--device", "cuda:0",
    ]

    # Run from ViewCrafter repo root (script uses relative imports)
    subprocess.run(cmd, check=True, cwd=str(VIEWCRAFTER_REPO))

    # ViewCrafter writes diffusion0.mp4 (or diffusion.mp4 depending on mode)
    exp_dir = out_dir / exp_name
    mp4 = None
    for cand in ("diffusion0.mp4", "diffusion.mp4"):
        if (exp_dir / cand).exists():
            mp4 = exp_dir / cand
            break
    if mp4 is None:
        raise RuntimeError(f"ViewCrafter produced no diffusion*.mp4 in {exp_dir}")

    frame_dir = exp_dir / "frames"
    return _extract_frames_from_mp4(mp4, frame_dir)


# ---------------------------------------------------------------------------
# Soft mask for synthetic frames (down-weights their L1 loss)
# ---------------------------------------------------------------------------

def _ensure_soft_mask(path: Path, value: int, width: int, height: int) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.full((height, width), int(value), dtype=np.uint8)
    cv2.imwrite(str(path), mask)


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------

def run(ctx: PipelineContext, params: dict, on_progress: ProgressCallback) -> None:
    cfg = settings.pipeline.viewcrafter
    if not cfg.enabled:
        return

    tf_path = ctx.ns_dir / "transforms.json"
    if not tf_path.exists():
        log.warning("[%s] transforms.json missing, skipping viewcrafter", ctx.job_id)
        return

    on_progress("viewcrafter", 44, "Detecting sparse keyframes...")

    tf = json.loads(tf_path.read_text())
    frames = tf.get("frames", [])
    if not frames:
        log.warning("[%s] No frames in transforms.json", ctx.job_id)
        return

    sparse = _find_sparse_keyframes(frames, cfg.sparse_radius_m, cfg.min_neighbors)
    sparse = sparse[: cfg.max_keyframes]
    if not sparse:
        log.info("[%s] No sparse keyframes (radius=%.2fm, min_n=%d) — skipping viewcrafter",
                 ctx.job_id, cfg.sparse_radius_m, cfg.min_neighbors)
        return

    log.info("[%s] ViewCrafter: %d sparse keyframes", ctx.job_id, len(sparse))
    on_progress("viewcrafter", 46, f"Generating {len(sparse)} orbit clips...")

    # Determine image dims from first real frame for soft mask sizing
    sample_img = ctx.frames_dir / Path(sparse[0]["file_path"]).name
    sample = cv2.imread(str(sample_img))
    if sample is None:
        log.warning("[%s] Could not read sample frame, skipping viewcrafter", ctx.job_id)
        return
    sh, sw = sample.shape[:2]

    soft_mask_rel = Path("masks") / "_soft_synthetic.png"
    _ensure_soft_mask(ctx.ns_dir / soft_mask_rel, cfg.synthetic_mask_value, sw, sh)

    syn_frames: list[dict] = []
    vc_scratch = ctx.out_dir / "viewcrafter"

    # Resolution per variant (naming convention: ViewCrafter_<frames>_<width>)
    vc_height, vc_width = (320, 512) if "512" in cfg.variant else (576, 1024)

    for kf in sparse:
        ref_img = ctx.frames_dir / Path(kf["file_path"]).name
        try:
            clip_paths = _call_viewcrafter(
                ref_image=ref_img,
                out_dir=vc_scratch / ref_img.stem,
                orbit_deg=cfg.orbit_deg,
                n_frames=cfg.frames_per_orbit,
                variant=cfg.variant,
                model_dir=Path(cfg.model_dir),
                dust3r_ckpt=Path(cfg.dust3r_ckpt),
                height=vc_height,
                width=vc_width,
            )
        except Exception as exc:
            log.warning("[%s] ViewCrafter failed for %s: %s", ctx.job_id, ref_img.name, exc)
            continue

        trajectory = _orbit_trajectory(len(clip_paths), cfg.orbit_deg)

        for i, frame_src in enumerate(clip_paths):
            syn_name = f"syn_{ref_img.stem}_{i:02d}.jpg"
            dst = ctx.ns_dir / "images" / syn_name
            # Re-encode as JPEG (ViewCrafter typically outputs PNG)
            img = cv2.imread(str(frame_src))
            if img is None:
                continue
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

            world_pose = _compose(kf["transform_matrix"], trajectory[i])
            syn_frames.append({
                "file_path": f"images/{syn_name}",
                "transform_matrix": world_pose.tolist(),
                "mask_path": str(soft_mask_rel),
            })

    if not syn_frames:
        log.warning("[%s] ViewCrafter produced no synthetic frames", ctx.job_id)
        return

    tf["frames"].extend(syn_frames)
    tf_path.write_text(json.dumps(tf, indent=2))

    on_progress("viewcrafter", 58, f"Added {len(syn_frames)} synthetic frames")
    log.info("[%s] ViewCrafter added %d synthetic frames from %d sparse keyframes",
             ctx.job_id, len(syn_frames), len(sparse))

    # Free VRAM before training
    try:
        import torch  # type: ignore
        torch.cuda.empty_cache()
    except Exception:
        pass
