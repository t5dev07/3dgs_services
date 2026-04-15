import shutil
from pathlib import Path
from typing import IO

from settings import settings as _settings


class LocalStorage:
    """Stores files on a shared Docker volume mounted at settings.storage.data_dir."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._root = data_dir or Path(_settings.storage.data_dir)
        (self._root / "uploads").mkdir(parents=True, exist_ok=True)
        (self._root / "outputs").mkdir(parents=True, exist_ok=True)

    def save_upload(self, key: str, stream: IO[bytes]) -> str:
        dest = self._root / key  # key already contains "uploads/" prefix
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            shutil.copyfileobj(stream, f)
        return key

    def get_path(self, key: str) -> Path:
        # key is relative: either "uploads/foo.mp4" or "outputs/job/model.ply"
        return self._root / key

    def get_download_url(self, key: str) -> str:
        # Served via /files/{key} route
        return f"/files/{key}"

    def exists(self, key: str) -> bool:
        return (self._root / key).exists()
