from celery import Celery

from settings import settings
from .celery_broker import CeleryBroker


def make_celery_app() -> Celery:
    return Celery(broker=settings.redis_broker_url)


def get_broker(app: Celery) -> CeleryBroker:
    return CeleryBroker(app)
