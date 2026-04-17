"""Pipeline orchestrator — runs all steps in order."""
import logging

from .context import PipelineContext, ProgressCallback
from .extract import get_quality_params
from . import extract, mask, colmap, depth, opensplat, postprocess, preview

log = logging.getLogger(__name__)


def run_pipeline(ctx: PipelineContext, on_progress: ProgressCallback) -> dict:
    """
    Run the full pipeline and return storage keys for outputs.

    Returns:
        {"ply_key": str | None, "splat_key": str | None, "preview_key": str | None}
    """
    params = get_quality_params(ctx.quality_preset)
    if ctx.iterations is not None:
        params["iterations"] = ctx.iterations

    # 1. Extract frames
    extract.run(ctx, params, on_progress)

    # 2. Person masking for COLMAP (non-fatal — skip if model missing)
    try:
        mask.run(ctx, on_progress)
    except Exception as exc:
        log.warning("[%s] Masking failed, continuing: %s", ctx.job_id, exc)

    # 3. COLMAP
    colmap.run(ctx, params, on_progress)

    # 4. Depth augmentation of COLMAP point cloud (non-fatal)
    try:
        depth.run(ctx, params, on_progress)
    except Exception as exc:
        log.warning("[%s] Depth augmentation failed, continuing: %s", ctx.job_id, exc)

    # 5. OpenSplat
    opensplat.run(ctx, params, on_progress)

    # 6. Post-process: opacity filter + .splat export (non-fatal)
    try:
        postprocess.run(ctx, on_progress)
    except Exception as exc:
        log.warning("[%s] Postprocess failed, continuing: %s", ctx.job_id, exc)

    # 7. Preview video (non-fatal)
    try:
        preview.run(ctx, on_progress)
    except Exception as exc:
        log.warning("[%s] Preview failed, continuing: %s", ctx.job_id, exc)

    return {
        "ply_key": f"outputs/{ctx.job_id}/model.ply" if ctx.ply_path.exists() else None,
        "splat_key": f"outputs/{ctx.job_id}/model.splat" if ctx.splat_path.exists() else None,
        "preview_key": f"outputs/{ctx.job_id}/preview.mp4" if ctx.preview_path.exists() else None,
    }
