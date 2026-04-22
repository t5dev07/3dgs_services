"""Nerfstudio 3DGS trainer — runs `ns-train <method>` with gsplat backend.

Replaces OpenSplat. Reads ctx.ns_dir/transforms.json (produced by transforms.py),
runs `ns-train <method> ...` then `ns-export gaussian-splat` → ctx.ply_path.

Supports:
  - splatfacto (stock Nerfstudio — scale reg + stop-split-at, no depth loss)
  - dn_splatter / dn_splatter_big (if installed — adds depth + normal supervision)

All tuning params are read from settings.pipeline.splatter / settings.pipeline.depth.
Depth-loss flags are only appended when method name starts with "dn_splatter".
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from collections import deque
from pathlib import Path

from settings import settings

from .context import PipelineContext, ProgressCallback

log = logging.getLogger(__name__)


def _run_streaming(cmd: list[str], log_file: Path) -> None:
    tail: deque[str] = deque(maxlen=60)
    with open(log_file, "a") as lf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            lf.write(line + "\n")
            tail.append(line)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(
            f"Command failed (rc={rc}): {' '.join(cmd[:3])} ...\nLast lines:\n" + "\n".join(tail)
        )


def _find_exported_ply(export_dir: Path) -> Path | None:
    """ns-export writes varying filenames across Nerfstudio versions."""
    for name in ("splat.ply", "point_cloud.ply", "gaussian_splat.ply"):
        cand = export_dir / name
        if cand.exists():
            return cand
    plys = list(export_dir.glob("*.ply"))
    return plys[0] if plys else None


def run(ctx: PipelineContext, params: dict, on_progress: ProgressCallback) -> None:
    cfg = settings.pipeline.splatter
    iterations = ctx.iterations if ctx.iterations is not None else int(params.get("iterations", 30000))
    on_progress("train", 45, f"Training {cfg.method} ({iterations} iterations)...")

    tf_path = ctx.ns_dir / "transforms.json"
    if not tf_path.exists():
        raise RuntimeError(f"transforms.json missing at {tf_path} — transforms step must run first")

    train_out = ctx.out_dir / "ns_train"
    train_out.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        "ns-train", cfg.method,
        "--data", str(ctx.ns_dir),
        "--output-dir", str(train_out),
        "--max-num-iterations", str(iterations),
        "--pipeline.model.use-scale-regularization", str(cfg.use_scale_regularization),
        "--pipeline.model.stop-split-at", str(cfg.stop_split_at),
        "--viewer.quit-on-train-completion", "True",
        "--vis", "tensorboard",
    ]
    # dn-splatter adds depth + normal supervision — only pass these flags when
    # the chosen method supports them. Stock splatfacto errors on unknown flags.
    if cfg.method.startswith("dn-splatter") or cfg.method.startswith("dn_splatter"):
        train_cmd += [
            "--pipeline.model.use-depth-loss", str(cfg.use_depth_loss),
            "--pipeline.model.depth-loss-type", cfg.depth_loss_type,
            "--pipeline.model.depth-lambda", str(cfg.depth_lambda),
            # We only supply depth maps (no GT normals). Derive pseudo-normals from
            # depth so normal loss has a valid target instead of None.
            "--pipeline.model.normal-supervision", cfg.normal_supervision,
            "--pipeline.model.use-normal-loss", str(cfg.use_normal_loss),
            "--pipeline.model.use-normal-tv-loss", str(cfg.use_normal_tv_loss),
            "--pipeline.model.normal-lambda", str(cfg.normal_lambda),
        ]
    # Dataparser + its flags must come after method flags
    train_cmd += [
        "nerfstudio-data",
        "--depth-unit-scale-factor", str(settings.pipeline.depth.unit_scale_factor),
    ]
    _run_streaming(train_cmd, ctx.log_file)

    # Locate config.yml produced by ns-train
    config_candidates = list(train_out.rglob("config.yml"))
    if not config_candidates:
        raise RuntimeError(f"ns-train finished but no config.yml found under {train_out}")
    config_path = max(config_candidates, key=lambda p: p.stat().st_mtime)
    log.info("[%s] Using config: %s", ctx.job_id, config_path)

    on_progress("train", 75, "Exporting Gaussian splat PLY...")
    export_dir = ctx.ply_path.parent / "ns_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    export_cmd = [
        "ns-export", "gaussian-splat",
        "--load-config", str(config_path),
        "--output-dir", str(export_dir),
    ]
    _run_streaming(export_cmd, ctx.log_file)

    exported = _find_exported_ply(export_dir)
    if exported is None:
        raise RuntimeError(f"ns-export produced no PLY in {export_dir}")

    shutil.move(str(exported), str(ctx.ply_path))

    size_mb = ctx.ply_path.stat().st_size / (1024 * 1024)
    on_progress("train", 78, f"Training complete — model {size_mb:.1f} MB")
    log.info("[%s] dn-splatter done: %s (%.1f MB)", ctx.job_id, ctx.ply_path, size_mb)
