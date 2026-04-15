"""OpenSplat Gaussian Splatting trainer — adapted from tmp_process_video.py."""
import logging
import os
import shutil
import subprocess
from collections import deque
from pathlib import Path

from .context import PipelineContext, ProgressCallback

log = logging.getLogger(__name__)


def _ensure_images(frames_dir: Path, sparse_dir: Path) -> None:
    """OpenSplat expects images under sparse_dir/images/."""
    images_dir = sparse_dir / "images"
    if images_dir.exists() and list(images_dir.glob("*.jpg")):
        return
    images_dir.mkdir(parents=True, exist_ok=True)
    for fp in frames_dir.glob("*.jpg"):
        dst = images_dir / fp.name
        if dst.exists():
            continue
        try:
            os.link(fp, dst)
        except OSError:
            shutil.copy2(fp, dst)
    log.info("Linked %d frames into %s", len(list(images_dir.glob("*.jpg"))), images_dir)


def run(ctx: PipelineContext, params: dict, on_progress: ProgressCallback) -> None:
    iterations = ctx.iterations if ctx.iterations is not None else int(params.get("iterations", 7000))
    on_progress("train", 45, f"Training OpenSplat ({iterations} iterations)...")

    sparse_dir = ctx.colmap_dir / "sparse" / "0"
    if not sparse_dir.exists():
        sparse_dir = ctx.colmap_dir / "sparse"

    _ensure_images(ctx.frames_dir, sparse_dir)

    cmd = [
        "opensplat",
        str(sparse_dir),
        "-o", str(ctx.ply_path),
        "-n", str(iterations),
        "--save-every", "-1",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    tail: deque[str] = deque(maxlen=40)
    with open(ctx.log_file, "a") as lf:
        if process.stdout:
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    lf.write(line + "\n")
                    tail.append(line)

    rc = process.wait()
    if rc != 0:
        raise RuntimeError(
            f"OpenSplat failed (rc={rc}). Last output:\n" + "\n".join(tail)
        )

    if not ctx.ply_path.exists():
        raise RuntimeError("OpenSplat finished but model.ply was not created")

    size_mb = ctx.ply_path.stat().st_size / (1024 * 1024)
    on_progress("train", 75, f"Training complete — model {size_mb:.1f} MB")
    log.info("[%s] OpenSplat done: %s (%.1f MB)", ctx.job_id, ctx.ply_path, size_mb)
