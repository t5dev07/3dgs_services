from typing import Protocol, runtime_checkable


@runtime_checkable
class BrokerBackend(Protocol):
    def submit(self, job_id: str) -> None:
        """Enqueue a job. Worker reads full metadata from JobStore."""
        ...
