import redis

from settings import settings
from .redis_store import RedisJobStore


def make_redis_client() -> redis.Redis:
    return redis.from_url(settings.redis_job_store_url, decode_responses=True)


def get_job_store(client: redis.Redis) -> RedisJobStore:
    return RedisJobStore(client)
