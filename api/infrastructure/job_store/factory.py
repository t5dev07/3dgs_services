import redis.asyncio as aioredis

from settings import settings
from .redis_store import RedisJobStore


def get_job_store(client: aioredis.Redis) -> RedisJobStore:
    return RedisJobStore(client)


def make_redis_client() -> aioredis.Redis:
    return aioredis.from_url(settings.redis_job_store_url, decode_responses=False)
