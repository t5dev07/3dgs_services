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


class DepthConfig(BaseModel):
    enabled: bool = True
    model_path: str = "/app/models/depth_anything_v2_vits.onnx"
    unit_scale_factor: float = 0.001   # mm → m (written into transforms.json)
    min_correspondences: int = 5        # min COLMAP 2D-3D matches to estimate scale
    skip_frames: int = 1                # process every Nth frame (1 = all)


class SplatterConfig(BaseModel):
    method: str = "dn-splatter"                 # dn-splatter | dn-splatter-big | splatfacto (CLI uses hyphens)
    use_scale_regularization: bool = True
    stop_split_at: int = 15000                  # iteration to stop Gaussian splitting
    ply_color_mode: str = "sh_coeffs"           # sh_coeffs | rgb
    # Depth loss fields — only applied when method is a dn_splatter variant.
    # Stock splatfacto ignores them (no such flags exist upstream).
    use_depth_loss: bool = True
    depth_loss_type: str = "LogL1"              # L1 | LogL1 | HuberL1 | MSE | DSSIML1
    depth_lambda: float = 0.2
    # Normal loss — dn-splatter enables normal loss by default but needs a GT
    # normal source. We don't provide mono-normal maps, so use "depth" supervision
    # to derive pseudo-normals from our depth maps.
    use_normal_loss: bool = True
    use_normal_tv_loss: bool = True
    normal_supervision: str = "depth"           # mono | depth
    normal_lambda: float = 0.1


class ViewCrafterConfig(BaseModel):
    enabled: bool = False
    model_dir: str = "/app/models/viewcrafter"
    dust3r_ckpt: str = "/app/models/viewcrafter/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    variant: str = "ViewCrafter_25_512"
    orbit_deg: float = 15.0
    frames_per_orbit: int = 16
    sparse_radius_m: float = 0.5
    min_neighbors: int = 3
    synthetic_mask_value: int = 128             # uint8, ≈0.5 soft mask
    max_keyframes: int = 20


class PipelineConfig(BaseModel):
    quality_preset: str = "balanced"
    iterations: Optional[int] = None  # None = use preset default
    depth: DepthConfig = DepthConfig()
    splatter: SplatterConfig = SplatterConfig()
    viewcrafter: ViewCrafterConfig = ViewCrafterConfig()


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
