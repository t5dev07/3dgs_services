from pathlib import Path

from settings import settings


class LocalStorage:
    def __init__(self, data_dir: Path | None = None) -> None:
        self._root = data_dir or Path(settings.storage.data_dir)

    def get_input_path(self, key: str) -> Path:
        return self._root / key

    def get_output_dir(self, job_id: str) -> Path:
        out = self._root / "outputs" / job_id
        out.mkdir(parents=True, exist_ok=True)
        return out

    def output_key(self, job_id: str, filename: str) -> str:
        return f"outputs/{job_id}/{filename}"
