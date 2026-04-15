"""Preview video — ffmpeg timelapse from selected frames."""
import logging
import subprocess

from .context import PipelineContext, ProgressCallback

log = logging.getLogger(__name__)


def run(ctx: PipelineContext, on_progress: ProgressCallback) -> None:
    on_progress("video", 90, "Generating preview video...")

    frames = sorted(ctx.frames_dir.glob("frame_*.jpg"))
    if not frames:
        log.warning("[%s] No frames found, skipping preview", ctx.job_id)
        return

    cmd = [
        "ffmpeg", "-y",
        "-framerate", "24",
        "-pattern_type", "glob", "-i", str(ctx.frames_dir / "*.jpg"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(ctx.preview_path),
    ]

    with open(ctx.log_file, "a") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        log.warning("[%s] Preview generation failed (rc=%d), continuing", ctx.job_id, result.returncode)
        return

    on_progress("video", 100, "Preview ready")
    log.info("[%s] Preview created: %s", ctx.job_id, ctx.preview_path)
