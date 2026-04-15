import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from settings import settings


def setup_logging() -> None:
    cfg = settings.logging
    Path(cfg.info_file).parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    info_h = RotatingFileHandler(cfg.info_file, maxBytes=cfg.max_bytes, backupCount=cfg.backup_count)
    info_h.setLevel(logging.INFO)
    info_h.setFormatter(fmt)

    err_h = RotatingFileHandler(cfg.error_file, maxBytes=cfg.max_bytes, backupCount=cfg.backup_count)
    err_h.setLevel(logging.ERROR)
    err_h.setFormatter(fmt)

    console_h = logging.StreamHandler()
    console_h.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(getattr(logging, cfg.level))
    root.addHandler(info_h)
    root.addHandler(err_h)
    root.addHandler(console_h)
