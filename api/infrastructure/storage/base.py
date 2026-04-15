from pathlib import Path
from typing import IO, Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    def save_upload(self, key: str, stream: IO[bytes]) -> str:
        """Persist an uploaded file. Returns the storage key."""
        ...

    def get_path(self, key: str) -> Path:
        """Resolve a storage key to an absolute local path (for FileResponse)."""
        ...

    def get_download_url(self, key: str) -> str:
        """Return a URL suitable for HTTP download (presigned or relative path)."""
        ...

    def exists(self, key: str) -> bool:
        ...
