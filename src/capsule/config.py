from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DOCUMENT_CHUNK_MAX_TOKENS = 400


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
    object_storage_public_endpoint: str | None = None
    object_storage_bucket: str = "capsule"
    object_storage_access_key: SecretStr = SecretStr("minioadmin")
    object_storage_secret_key: SecretStr = SecretStr("minioadmin")
    object_storage_region: str = "us-east-1"
    import_root: Path = Path("data/imports")
    import_file_max_bytes: int = Field(default=2 * 1024 * 1024 * 1024, ge=1)
    assetization_version: str = Field(default="assetization-v3", min_length=1, max_length=64)

    ark_api_key: SecretStr | None = None
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    understanding_model: str = "doubao-seed-2-0-lite-260215"
    embedding_model: str = "doubao-embedding-vision-250615"
    embedding_dimension: int = Field(default=1024, ge=1)
    tokenization_batch_size: int = Field(default=64, ge=1, le=256)
    document_chunk_max_tokens: int = Field(default=DOCUMENT_CHUNK_MAX_TOKENS, ge=1)

    understanding_concurrency: int = Field(default=32, ge=1)
    embedding_concurrency: int = Field(default=64, ge=1)
    search_embedding_concurrency: int = Field(default=16, ge=1)
    capsule_concurrency: int = Field(default=8, ge=1)
    file_parse_concurrency: int = Field(default=4, ge=1)
    ffmpeg_concurrency: int = Field(default=2, ge=1)
    video_scene_threshold: float = Field(default=27.0, gt=0)
    video_min_scene_seconds: float = Field(default=1.0, gt=0)
    video_max_shot_seconds: float = Field(default=45.0, gt=0)
    video_window_seconds: float = Field(default=20.0, gt=0)
    video_sample_interval_seconds: float = Field(default=5.0, gt=0)
    video_max_candidate_frames: int = Field(default=12, ge=2)
    video_max_representative_frames: int = Field(default=3, ge=1, le=3)
    mobileclip_model_path: str = "data/models/mobileclip-s0/mobileclip_s0.pt"
    mobileclip_batch_size: int = Field(default=12, ge=1)
    model_max_retries: int = Field(default=4, ge=1, le=10)
    understanding_timeout_seconds: float = Field(default=180.0, gt=0)
    embedding_timeout_seconds: float = Field(default=60.0, gt=0)
    milvus_batch_size: int = Field(default=100, ge=1)

    search_channel_top_k_multiplier: int = Field(default=3, ge=1)
    search_channel_top_k_cap: int = Field(default=100, ge=1)
    search_candidate_cap: int = Field(default=300, ge=1)
    search_same_source_limit: int = Field(default=3, ge=1)
    search_rrf_k: int = Field(default=60, ge=1)
    search_hnsw_ef: int = Field(default=128, ge=1)
    search_engine_version: str = "search-v1"
    search_capsule_recent_limit: int = Field(default=10, ge=1)
    query_image_max_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    search_native_weight: float = Field(default=1.0, gt=0)
    search_description_weight: float = Field(default=0.8, gt=0)
    search_subject_weight: float = Field(default=0.8, gt=0)
    search_visual_weight: float = Field(default=0.6, gt=0)
    search_cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "https://capsule-search-workspace.ranhaoxu212.chatgpt.site",
        ]
    )

    @field_validator("ark_api_key", "milvus_token", mode="before")
    @classmethod
    def empty_secret_is_unset(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
