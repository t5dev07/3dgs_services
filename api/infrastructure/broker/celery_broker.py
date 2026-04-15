import logging

from celery import Celery

log = logging.getLogger(__name__)

_TASK_NAME = "run_pipeline"


class CeleryBroker:
    def __init__(self, app: Celery) -> None:
        self._app = app

    def submit(self, job_id: str) -> None:
        self._app.send_task(_TASK_NAME, args=[job_id])
        log.info("Job submitted to broker: %s", job_id)
