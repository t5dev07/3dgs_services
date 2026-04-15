from typing import Protocol, runtime_checkable


@runtime_checkable
class JobStore(Protocol):
    def update(self, job_id: str, **fields) -> None:
        """Partial update job fields in the store."""
        ...
