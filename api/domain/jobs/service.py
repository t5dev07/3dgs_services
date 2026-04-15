import logging
import uuid
from datetime import datetime
from typing import IO, Optional

from domain.jobs.entities import Job, JobStage, JobStatus
from infrastructure.broker.base import BrokerBackend
from infrastructure.job_store.base import JobStore
from infrastructure.storage.base import StorageBackend

log = logging.getLogger(__name__)


class JobService:
    def __init__(
        self,
        job_store: JobStore,
        broker: BrokerBackend,
        storage: StorageBackend,
    ) -> None:
        self._store = job_store
        self._broker = broker
        self._storage = storage

    async def create_from_upload(
        self,
        filename: str,
        stream: IO[bytes],
    ) -> Job:
        job_id = uuid.uuid4().hex[:12]
        input_key = f"uploads/{job_id}_{filename}"

        self._storage.save_upload(input_key, stream)

        job = Job(
            id=job_id,
            status=JobStatus.QUEUED,
            stage=JobStage.QUEUED,
            input_key=input_key,
            message="Job queued",
            created_at=datetime.utcnow(),
        )
        await self._store.create(job)
        self._broker.submit(job_id)

        log.info("Job created from upload: %s (file=%s)", job_id, filename)
        return job

    async def get(self, job_id: str) -> Optional[Job]:
        return await self._store.get(job_id)

    async def list_all(self) -> list[Job]:
        return await self._store.list_all()

    def get_download_path(self, key: str):
        return self._storage.get_path(key)

    def get_download_url(self, key: str) -> str:
        return self._storage.get_download_url(key)

    def storage_exists(self, key: str) -> bool:
        return self._storage.exists(key)
