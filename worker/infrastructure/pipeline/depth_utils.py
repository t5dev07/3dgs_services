"""Shared COLMAP text-format parsers used by depth + viewcrafter steps."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class Camera:
    __slots__ = ("model", "width", "height", "fx", "fy", "cx", "cy")

    def __init__(self, model: str, width: int, height: int, params: list[str]):
        self.model = model
        self.width = width
        self.height = height
        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
            self.fx = self.fy = float(params[0])
            self.cx = float(params[1])
            self.cy = float(params[2])
        elif model in ("PINHOLE", "OPENCV", "FULL_OPENCV", "RADIAL"):
            self.fx = float(params[0])
            self.fy = float(params[1])
            self.cx = float(params[2])
            self.cy = float(params[3])
        else:
            self.fx = self.fy = float(params[0])
            self.cx = width / 2.0
            self.cy = height / 2.0


class Image:
    __slots__ = ("image_id", "qvec", "tvec", "camera_id", "name", "observations")

    def __init__(self, image_id, qvec, tvec, camera_id, name, observations):
        self.image_id = image_id
        self.qvec = qvec          # (qw, qx, qy, qz)
        self.tvec = tvec          # (tx, ty, tz)
        self.camera_id = camera_id
        self.name = name
        self.observations = observations  # list of (u, v, point3d_id)

    def rotation_matrix(self) -> np.ndarray:
        qw, qx, qy, qz = self.qvec
        return np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz),     2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy),     2 * (qy * qz + qw * qx),     1 - 2 * (qx * qx + qy * qy)],
        ], dtype=np.float64)

    def translation(self) -> np.ndarray:
        return np.array(self.tvec, dtype=np.float64)


def find_best_sparse_dir(colmap_dir: Path) -> Optional[Path]:
    """Return the largest COLMAP sub-reconstruction under `colmap_dir/sparse/`.

    COLMAP mapper emits `sparse/0`, `sparse/1`, ... in creation order — NOT by
    size. For fragmented walkthrough videos `sparse/0` may be a 2-frame stub
    while the main reconstruction is in `sparse/1` or `sparse/2`. Size is
    approximated by `images.bin` file size (proportional to registered frames).
    Falls back to `sparse/` if no numbered subdir exists.
    """
    sparse_root = colmap_dir / "sparse"
    if not sparse_root.exists():
        return None

    subdirs = [d for d in sparse_root.iterdir() if d.is_dir() and (d / "images.bin").exists()]
    if subdirs:
        return max(subdirs, key=lambda d: (d / "images.bin").stat().st_size)
    if (sparse_root / "images.bin").exists():
        return sparse_root
    return None


def colmap_model_converter(src: Path, dst: Path, output_type: str, log_file: Path) -> bool:
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


def parse_cameras_txt(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            cameras[cam_id] = Camera(
                model=parts[1],
                width=int(parts[2]),
                height=int(parts[3]),
                params=parts[4:],
            )
    return cameras


def parse_images_txt(path: Path) -> list[Image]:
    images = []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        image_id = int(parts[0])
        qvec = tuple(float(x) for x in parts[1:5])
        tvec = tuple(float(x) for x in parts[5:8])
        camera_id = int(parts[8])
        name = parts[9]
        obs = []
        if i + 1 < len(lines):
            obs_parts = lines[i + 1].split()
            for j in range(0, len(obs_parts) - 2, 3):
                u = float(obs_parts[j])
                v = float(obs_parts[j + 1])
                pt_id = int(obs_parts[j + 2])
                obs.append((u, v, pt_id))
        images.append(Image(image_id, qvec, tvec, camera_id, name, obs))
        i += 2
    return images


def parse_points3d_txt(path: Path) -> dict[int, np.ndarray]:
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
