import logging
from datetime import datetime
from typing import Optional

import redis.asyncio as aioredis

from domain.jobs.entities import Job

log = logging.getLogger(__name__)

_JOB_PREFIX = "job:"
_INDEX_KEY = "jobs:index"  # sorted set: score=created_at timestamp, member=job_id


class RedisJobStore:
    def __init__(self, client: aioredis.Redis) -> None:
        self._r = client

    async def create(self, job: Job) -> None:
        data = job.to_dict()
        await self._r.hset(f"{_JOB_PREFIX}{job.id}", mapping=data)
        await self._r.zadd(_INDEX_KEY, {job.id: job.created_at.timestamp()})
        log.info("Job created: %s", job.id)

    async def get(self, job_id: str) -> Optional[Job]:
        data = await self._r.hgetall(f"{_JOB_PREFIX}{job_id}")
        if not data:
            return None
        # Redis returns bytes — decode
        decoded = {k.decode(): v.decode() for k, v in data.items()}
        return Job.from_dict(decoded)

    async def update(self, job_id: str, **fields) -> None:
        if not fields:
            return
        fields["updated_at"] = datetime.utcnow().isoformat()
        # Coerce values to str for Redis hash storage
        mapping = {k: str(v) if v is not None else "" for k, v in fields.items()}
        await self._r.hset(f"{_JOB_PREFIX}{job_id}", mapping=mapping)

    async def list_all(self) -> list[Job]:
        job_ids = await self._r.zrevrange(_INDEX_KEY, 0, -1)
        jobs = []
        for jid in job_ids:
            job = await self.get(jid.decode())
            if job:
                jobs.append(job)
        return jobs
