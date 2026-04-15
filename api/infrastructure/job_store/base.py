from typing import Optional, Protocol, runtime_checkable

from domain.jobs.entities import Job


@runtime_checkable
class JobStore(Protocol):
    async def create(self, job: Job) -> None:
        ...

    async def get(self, job_id: str) -> Optional[Job]:
        ...

    async def update(self, job_id: str, **fields) -> None:
        """Partial update — only provided fields are changed."""
        ...

    async def list_all(self) -> list[Job]:
        ...
