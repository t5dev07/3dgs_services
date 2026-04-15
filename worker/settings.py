import os
from typing import Optional

from pydantic import BaseModel, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource
from pydantic_settings.main import PydanticBaseSettingsSource


class RedisConfig(BaseModel):
    url: str = "redis://redis:6379"
    broker_db: int = 0
    job_store_db: int = 1


class StorageConfig(BaseModel):
    backend: str = "local"
    data_dir: str = "/data"


class PipelineConfig(BaseModel):
    quality_preset: str = "balanced"
    iterations: Optional[int] = None  # None = use preset default


class LoggingConfig(BaseModel):
    level: str = "INFO"
    max_bytes: int = 10_485_760
    backup_count: int = 5
    info_file: str = "/data/logs/worker.log"
    error_file: str = "/data/logs/worker-error.log"


class Settings(BaseSettings):
    redis: RedisConfig = RedisConfig()
    storage: StorageConfig = StorageConfig()
    pipeline: PipelineConfig = PipelineConfig()
    logging: LoggingConfig = LoggingConfig()

    redis_password: str = ""

    model_config = SettingsConfigDict(
        yaml_file=os.getenv("SETTINGS_PATH", "/config/settings.yaml"),
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Optional[PydanticBaseSettingsSource] = None,
        env_settings: Optional[PydanticBaseSettingsSource] = None,
        dotenv_settings: Optional[PydanticBaseSettingsSource] = None,
        secrets_dir_settings: Optional[PydanticBaseSettingsSource] = None,
        **_: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources = [init_settings, env_settings, dotenv_settings, YamlConfigSettingsSource(settings_cls)]
        return tuple(s for s in sources if s is not None)  # type: ignore[misc]

    @computed_field  # type: ignore[misc]
    @property
    def redis_broker_url(self) -> str:
        return self._build_redis_url(self.redis.broker_db)

    @computed_field  # type: ignore[misc]
    @property
    def redis_job_store_url(self) -> str:
        return self._build_redis_url(self.redis.job_store_db)

    def _build_redis_url(self, db: int) -> str:
        base = self.redis.url
        if self.redis_password:
            base = base.replace("redis://", f"redis://:{self.redis_password}@", 1)
        return f"{base}/{db}"


settings = Settings()
