"""FastAPI dependency injection — all per-request deps are assembled here."""
from functools import lru_cache

import redis.asyncio as aioredis
from celery import Celery
from fastapi import Request

from domain.jobs.service import JobService
from infrastructure.broker.factory import get_broker, make_celery_app
from infrastructure.job_store.factory import get_job_store, make_redis_client
from infrastructure.storage.factory import get_storage


@lru_cache
def _celery_app() -> Celery:
    return make_celery_app()


def get_redis(request: Request) -> aioredis.Redis:
    return request.app.state.redis


def get_job_service(request: Request) -> JobService:
    redis_client = get_redis(request)
    return JobService(
        job_store=get_job_store(redis_client),
        broker=get_broker(_celery_app()),
        storage=get_storage(),
    )
