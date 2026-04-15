import logging
from datetime import datetime

import redis

log = logging.getLogger(__name__)

_JOB_PREFIX = "job:"


class RedisJobStore:
    def __init__(self, client: redis.Redis) -> None:
        self._r = client

    def update(self, job_id: str, **fields) -> None:
        if not fields:
            return
        fields["updated_at"] = datetime.utcnow().isoformat()
        mapping = {k: str(v) if v is not None else "" for k, v in fields.items()}
        self._r.hset(f"{_JOB_PREFIX}{job_id}", mapping=mapping)
        log.debug("Job %s updated: %s", job_id, list(fields.keys()))
