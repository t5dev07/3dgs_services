from settings import settings
from .local import LocalStorage


def get_storage() -> LocalStorage:
    if settings.storage.backend == "local":
        return LocalStorage()
    raise ValueError(f"Unknown storage backend: {settings.storage.backend!r}")
