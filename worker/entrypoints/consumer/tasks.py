import logging

from celery import Celery

from logging_config import setup_logging
from settings import settings
from infrastructure.job_store.factory import get_job_store, make_redis_client
from infrastructure.storage.factory import get_storage
from infrastructure.pipeline import run_pipeline
from infrastructure.pipeline.context import PipelineContext

setup_logging()
log = logging.getLogger(__name__)

app = Celery("worker", broker=settings.redis_broker_url)

# Shared Redis client (reused across tasks in the same worker process)
_redis = make_redis_client()


@app.task(name="run_pipeline", bind=True, max_retries=0)
def handle_job(self, job_id: str) -> None:
    store = get_job_store(_redis)
    storage = get_storage()

    log.info("Received job: %s", job_id)

    try:
        store.update(job_id, status="processing", stage="queued", progress=0, message="Worker picked up job")

        # Resolve paths from storage
        # input_key was stored in JobStore when the API created the job
        job_data = _redis.hgetall(f"job:{job_id}")
        input_key = job_data.get("input_key", "")
        if not input_key:
            raise RuntimeError(f"No input_key found for job {job_id}")

        video_path = storage.get_input_path(input_key)
        if not video_path.exists():
            raise RuntimeError(f"Input video not found at {video_path}")

        out_dir = storage.get_output_dir(job_id)

        ctx = PipelineContext(
            job_id=job_id,
            video_path=video_path,
            out_dir=out_dir,
            quality_preset=settings.pipeline.quality_preset,
            iterations=settings.pipeline.iterations,
        )

        def on_progress(stage: str, progress: int, message: str = "") -> None:
            store.update(job_id, stage=stage, progress=progress, message=message)

        result = run_pipeline(ctx, on_progress)

        store.update(
            job_id,
            status="done",
            stage="done",
            progress=100,
            message="Pipeline complete",
            ply_key=result.get("ply_key") or "",
            preview_key=result.get("preview_key") or "",
        )
        log.info("Job %s completed", job_id)

    except Exception as exc:
        log.exception("Job %s failed", job_id)
        store.update(job_id, status="failed", message="Pipeline failed", error=str(exc))
        raise
