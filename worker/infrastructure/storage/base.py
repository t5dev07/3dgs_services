from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    def get_input_path(self, key: str) -> Path:
        """Resolve upload key to absolute local path."""
        ...

    def get_output_dir(self, job_id: str) -> Path:
        """Return the output directory for a job."""
        ...

    def output_key(self, job_id: str, filename: str) -> str:
        """Build a storage key for a job output file."""
        ...
