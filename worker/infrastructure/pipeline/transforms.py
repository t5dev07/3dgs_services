"""Convert COLMAP reconstruction → Nerfstudio transforms.json.

Output layout (ctx.ns_dir):
    ns_data/
      transforms.json     # frames with file_path + depth_file_path + mask_path
      images/             # symlinks to ctx.frames_dir/*.jpg
      depths/  (symlink)  # → ctx.depths_dir
      masks/   (symlink)  # → ctx.masks_dir

dn-splatter (via ns-train --data ns_data) reads transforms.json with the
`nerfstudio-data` dataparser, which consumes depth_file_path + mask_path fields.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from settings import settings

from .context import PipelineContext, ProgressCallback
from .depth_utils import find_best_sparse_dir

log = logging.getLogger(__name__)


def _symlink_dir(src: Path, link: Path) -> None:
    """Create a directory symlink link → src. Idempotent."""
    if link.is_symlink() or link.exists():
        try:
            if link.is_symlink() or link.is_file():
                link.unlink()
            else:
                shutil.rmtree(link)
        except OSError:
            pass
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src, link, target_is_directory=True)


def _link_images(frames_dir: Path, images_dir: Path) -> int:
    images_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for fp in frames_dir.glob("*.jpg"):
        dst = images_dir / fp.name
        if dst.exists():
            count += 1
            continue
        try:
            os.link(fp, dst)
        except OSError:
            shutil.copy2(fp, dst)
        count += 1
    return count


def run(ctx: PipelineContext, params: dict, on_progress: ProgressCallback) -> None:
    cfg = settings.pipeline.depth
    on_progress("transforms", 42, "Writing transforms.json...")

    sparse_dir = find_best_sparse_dir(ctx.colmap_dir)
    if sparse_dir is None:
        raise RuntimeError("No COLMAP sparse reconstruction found for transforms.json")
    log.info("[%s] Using sparse reconstruction: %s", ctx.job_id, sparse_dir)

    # Use Nerfstudio's built-in converter
    try:
        from nerfstudio.process_data.colmap_utils import colmap_to_json
    except ImportError as exc:
        raise RuntimeError(f"nerfstudio not installed in worker env: {exc}")

    ctx.ns_dir.mkdir(parents=True, exist_ok=True)

    # Signature across Nerfstudio versions: colmap_to_json(recon_dir, output_dir, ...)
    # Newer versions accept image_rename_map=None and return number of frames.
    try:
        colmap_to_json(recon_dir=sparse_dir, output_dir=ctx.ns_dir)
    except TypeError:
        # Some versions use different kwarg names — fall back to positional
        colmap_to_json(sparse_dir, ctx.ns_dir)

    tf_path = ctx.ns_dir / "transforms.json"
    if not tf_path.exists():
        raise RuntimeError(f"colmap_to_json did not produce {tf_path}")

    data = json.loads(tf_path.read_text())
    data["depth_unit_scale_factor"] = cfg.unit_scale_factor

    # Link data dirs so mask_path / depth_file_path resolve relative to ns_dir
    _link_images(ctx.frames_dir, ctx.ns_dir / "images")
    if ctx.depths_dir.exists():
        _symlink_dir(ctx.depths_dir, ctx.ns_dir / "depths")
    if ctx.masks_dir.exists():
        _symlink_dir(ctx.masks_dir, ctx.ns_dir / "masks")

    # Attach depth_file_path per frame. Masks intentionally omitted: Nerfstudio's
    # full_images_datamanager undistorts RGB but not masks, producing off-by-one
    # shape mismatches in dn_splatter eval (gt_rgb * mask → tensor size mismatch).
    # YOLO person-masks cover <2% of frames anyway — dropped until undistortion
    # is handled end-to-end.
    depths_attached = 0

    for frame in data["frames"]:
        stem = Path(frame["file_path"]).stem

        depth_rel = Path("depths") / f"{stem}.png"
        if (ctx.ns_dir / depth_rel).exists():
            frame["depth_file_path"] = str(depth_rel)
            depths_attached += 1

        frame.pop("mask_path", None)

    tf_path.write_text(json.dumps(data, indent=2))

    on_progress(
        "transforms", 44,
        f"transforms.json: {len(data['frames'])} frames, {depths_attached} depth",
    )
    log.info(
        "[%s] transforms.json: %d frames, %d depth",
        ctx.job_id, len(data["frames"]), depths_attached,
    )
