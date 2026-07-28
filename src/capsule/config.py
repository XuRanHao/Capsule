from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CAPSULE_",
        extra="ignore",
    )

    env: str = "development"
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://capsule:capsule@localhost:5432/capsule"
    milvus_uri: str = "http://localhost:19530"
    milvus_token: SecretStr | None = None
    milvus_collection: str = "asset_embeddings_seed16_1024"

    object_storage_endpoint: str = "http://localhost:9000"
    object_storage_bucket: str = "capsule"
    object_storage_access_key: SecretStr = SecretStr("minioadmin")
    object_storage_secret_key: SecretStr = SecretStr("minioadmin")
    object_storage_region: str = "us-east-1"

    ark_api_key: SecretStr | None = None
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    understanding_model: str = "doubao-seed-2-0-lite-260215"
    embedding_model: str = "doubao-embedding-vision-250615"
    embedding_dimension: int = Field(default=1024, ge=1)

    understanding_concurrency: int = Field(default=6, ge=1)
    embedding_concurrency: int = Field(default=16, ge=1)
    capsule_concurrency: int = Field(default=4, ge=1)
    file_parse_concurrency: int = Field(default=4, ge=1)
    ffmpeg_concurrency: int = Field(default=2, ge=1)
    model_max_retries: int = Field(default=4, ge=1, le=10)
    understanding_timeout_seconds: float = Field(default=180.0, gt=0)
    embedding_timeout_seconds: float = Field(default=60.0, gt=0)
    milvus_batch_size: int = Field(default=100, ge=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
