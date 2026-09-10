from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from capsule.video_output import VideoOutputMode

DOCUMENT_CHUNK_MIN_TOKENS = 250
DOCUMENT_CHUNK_TARGET_TOKENS = 400
DOCUMENT_CHUNK_MAX_TOKENS = 500
DOCUMENT_CHUNK_MERGE_MAX_TOKENS = 600


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
    milvus_fused_search_collection: str = "asset_embeddings_seed16_1024_fused_n03_v1"

    object_storage_endpoint: str = "http://localhost:9000"
    object_storage_public_endpoint: str | None = None
    object_storage_bucket: str = "capsule"
    object_storage_access_key: SecretStr = SecretStr("minioadmin")
    object_storage_secret_key: SecretStr = SecretStr("minioadmin")
    object_storage_region: str = "us-east-1"
    import_root: Path = Path("data/imports")
    import_file_max_bytes: int = Field(default=2 * 1024 * 1024 * 1024, ge=1)
    assetization_version: str = Field(default="assetization-v6", min_length=1, max_length=64)

    ark_api_key: SecretStr | None = None
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    understanding_model: str = "doubao-seed-2-0-lite-260428"
    search_query_model: str = Field(
        default="deepseek-v4-flash",
        validation_alias=AliasChoices(
            "search_query_model",
            "CAPSULE_SEARCH_QUERY_MODEL",
            "CAPSULE_SEARCH_WEIGHT_MODEL",
            "CAPSULE_SEARCH_PARSER_MODEL",
        ),
    )
    embedding_model: str = "doubao-embedding-vision-250615"
    embedding_dimension: int = Field(default=1024, ge=1)
    tokenization_batch_size: int = Field(default=64, ge=1, le=256)
    document_tokenizer_path: Path | None = None
    document_chunk_min_tokens: int = Field(default=DOCUMENT_CHUNK_MIN_TOKENS, ge=1)
    document_chunk_target_tokens: int = Field(default=DOCUMENT_CHUNK_TARGET_TOKENS, ge=1)
    document_chunk_max_tokens: int = Field(default=DOCUMENT_CHUNK_MAX_TOKENS, ge=1)
    document_chunk_merge_max_tokens: int = Field(
        default=DOCUMENT_CHUNK_MERGE_MAX_TOKENS,
        ge=1,
    )
    document_parent_max_tokens: int = Field(default=2000, ge=1)
    document_media_root: Path = Path("data/document-media")
    document_ocr_enabled: bool = True
    document_ocr_min_confidence: float = Field(default=0.55, ge=0, le=1)
    document_ocr_min_edge: int = Field(default=32, ge=1)
    document_ocr_min_area: int = Field(default=4096, ge=1)

    understanding_concurrency: int = Field(default=32, ge=1)
    asset_enrichment_queue_size: int = Field(default=64, ge=1)
    search_understanding_concurrency: int = Field(default=4, ge=1)
    native_embedding_concurrency: int = Field(default=24, ge=1)
    video_native_embedding_concurrency: int = Field(default=2, ge=1)
    embedding_concurrency: int = Field(default=96, ge=1)
    search_embedding_concurrency: int = Field(default=16, ge=1)
    capsule_concurrency: int = Field(default=8, ge=1)
    http_max_connections: int = Field(default=192, ge=1)
    http_max_keepalive_connections: int = Field(default=128, ge=1)
    model_image_target_bytes: int = Field(default=2 * 1024 * 1024, ge=1)
    model_image_max_edge: int = Field(default=1536, ge=1)
    understanding_image_size: int = Field(default=768, ge=224, le=1536)
    model_image_cache_entries: int = Field(default=128, ge=1)
    file_parse_concurrency: int = Field(default=4, ge=1)
    ffmpeg_concurrency: int = Field(default=2, ge=1)
    video_upload_concurrency: int = Field(default=4, ge=1)
    video_spool_root: Path = Path("data/video-spool")
    video_spool_max_items: int = Field(default=32, ge=1)
    video_spool_max_bytes: int = Field(default=4 * 1024 * 1024 * 1024, ge=1)
    video_upload_queue_backend: Literal["memory", "redis"] = "redis"
    redis_url: str = "redis://localhost:6379/0"
    video_upload_stream: str = "capsule:video-uploads"
    video_upload_group: str = "capsule-video-uploaders"
    video_upload_claim_idle_ms: int = Field(default=30_000, ge=100)
    video_upload_max_attempts: int = Field(default=4, ge=1, le=20)
    video_upload_retry_base_seconds: float = Field(default=0.5, ge=0, le=30)
    # Durable whole-video task transport. PostgreSQL is authoritative; these
    # settings only control Streams delivery, lease recovery, and polling.
    video_task_stream: str = "capsule:video:tasks"
    video_task_group: str = "capsule-video-workers"
    video_task_dlq_stream: str = "capsule:video:dlq"
    video_task_claim_idle_ms: int = Field(default=30_000, ge=100)
    video_task_lease_seconds: float = Field(default=60.0, gt=0)
    video_task_heartbeat_seconds: float = Field(default=10.0, gt=0)
    video_task_progress_timeout_seconds: float = Field(default=120.0, gt=0)
    video_task_hard_timeout_seconds: float = Field(default=7_200.0, gt=0)
    video_task_redispatch_seconds: float = Field(default=60.0, gt=0)
    video_task_redis_max_failures: int = Field(default=4, ge=1, le=20)
    video_task_max_dispatch_rounds: int = Field(default=4, ge=1, le=20)
    video_task_retry_delays_seconds: list[float] = Field(
        default_factory=lambda: [5.0, 30.0, 300.0],
        min_length=1,
    )
    video_task_scheduler_poll_seconds: float = Field(default=5.0, gt=0)
    # Durable CPU image/text task routes. Each kind owns an independent stream
    # and consumer group so a worker can only receive work its repository is
    # authorized to claim.
    cpu_image_task_stream: str = "capsule:image:tasks"
    cpu_image_task_group: str = "capsule-image-workers"
    cpu_image_task_dlq_stream: str = "capsule:image:dlq"
    cpu_image_task_concurrency: int = Field(default=16, ge=1)
    cpu_text_task_stream: str = "capsule:text:tasks"
    cpu_text_task_group: str = "capsule-text-workers"
    cpu_text_task_dlq_stream: str = "capsule:text:dlq"
    cpu_text_task_concurrency: int = Field(default=4, ge=1)
    cpu_source_roots: list[Path] = Field(default_factory=list)
    # A single-process/local API must be able to consume the durable tasks it
    # creates. Production deployments with dedicated worker services can turn
    # this off and run the worker/scheduler CLI commands independently.
    api_embedded_cpu_tasks_enabled: bool = True
    api_embedded_cpu_tasks_restart_seconds: float = Field(default=2.0, gt=0)
    api_embedded_cpu_tasks_startup_timeout_seconds: float = Field(default=30.0, gt=0)
    # ``materialized`` preserves the legacy playable MP4/keyframe artifacts.
    # ``logical`` stores only content-aware ranges and representative timestamps;
    # Assets still run through understanding, vectorization, and clustering.
    video_output_mode: VideoOutputMode = "logical"
    # Additional local roots permitted for durable video source references.
    # They are checked before a file:// URI reaches PostgreSQL or Redis.
    video_source_roots: list[Path] = Field(default_factory=list)
    video_transcode_concurrency: int = Field(default=1, ge=1, le=8)
    video_sample_interval_seconds: float = Field(default=0.5, gt=0)
    video_min_segment_seconds: float = Field(default=1.0, gt=0)
    video_distance_quantile: float = Field(default=0.75, ge=0, le=1)
    video_min_distance_threshold: float = Field(default=0.08, ge=0, le=2)
    video_max_distance_threshold: float = Field(default=0.25, ge=0, le=2)
    video_similarity_relaxation: float = Field(default=0.05, ge=0, le=2)
    video_max_merge_cost: float = Field(default=0.5, ge=0)
    video_base_target_seconds: float = Field(default=10.0, gt=0)
    video_max_target_seconds: float = Field(default=20.0, gt=0)
    video_target_log2_weight: float = Field(default=0.15, ge=0)
    video_hard_max_duration_factor: float = Field(default=1.5, ge=1)
    video_activity_sample_fps: float = Field(default=6.0, gt=0)
    video_activity_envelope_seconds: float = Field(default=0.5, gt=0)
    video_activity_shift_side_seconds: float = Field(default=1.0, gt=0)
    video_keyframe_size: int = Field(default=224, ge=224, le=224)
    video_keyframe_jpeg_quality: int = Field(default=85, ge=1, le=100)
    video_max_representative_frames: int = Field(default=3, ge=1, le=3)
    mobileclip_model_path: str = "data/models/mobileclip-s0/mobileclip_s0.pt"
    mobileclip_batch_size: int = Field(default=12, ge=1)
    audio_window_seconds: float = Field(default=1.0, gt=0)
    audio_min_segment_seconds: float = Field(default=3.0, gt=0)
    audio_sample_rate: int = Field(default=32_000, ge=8_000, le=96_000)
    audio_distance_quantile: float = Field(default=0.75, ge=0, le=1)
    audio_min_distance_threshold: float = Field(default=0.002, ge=0, le=2)
    audio_max_distance_threshold: float = Field(default=0.02, ge=0, le=2)
    efficientat_source_path: Path = Path("data/models/EfficientAT")
    efficientat_model_name: str = "mn10_as"
    efficientat_batch_size: int = Field(default=16, ge=1)
    model_max_retries: int = Field(default=4, ge=1, le=10)
    understanding_timeout_seconds: float = Field(default=180.0, gt=0)
    understanding_max_output_tokens: int = Field(default=2048, ge=256)
    search_query_max_output_tokens: int = Field(
        default=500,
        ge=128,
        le=1024,
        validation_alias=AliasChoices(
            "search_query_max_output_tokens",
            "CAPSULE_SEARCH_QUERY_MAX_OUTPUT_TOKENS",
            "CAPSULE_SEARCH_WEIGHT_MAX_OUTPUT_TOKENS",
            "CAPSULE_SEARCH_PARSER_MAX_OUTPUT_TOKENS",
        ),
    )
    embedding_timeout_seconds: float = Field(default=60.0, gt=0)
    milvus_batch_size: int = Field(default=100, ge=1)

    cluster_selection_epsilon: float = Field(default=0.5, ge=0.0, le=2.0)
    cluster_incremental_assignment_threshold: float = Field(
        default=0.88,
        ge=-1.0,
        le=1.0,
    )
    cluster_bootstrap_minimum_count: int = Field(default=50, ge=1)
    cluster_bootstrap_concurrency: int = Field(default=1, ge=1)
    cluster_auto_recluster_new_ratio: float = Field(default=0.3, gt=0.0, le=1.0)
    cluster_auto_recluster_minimum_new_count: int = Field(default=20, ge=1)

    search_channel_top_k_multiplier: int = Field(default=3, ge=1)
    search_channel_top_k_cap: int = Field(default=100, ge=1)
    search_candidate_cap: int = Field(default=300, ge=1)
    search_same_source_limit: int = Field(default=3, ge=1)
    search_cluster_top_k: int = Field(default=12, ge=1, le=100)
    search_rrf_k: int = Field(default=20, ge=1)
    search_hnsw_ef: int = Field(default=128, ge=1)
    search_vector_visibility_timeout_seconds: float = Field(default=5.0, ge=0)
    search_vector_visibility_poll_initial_seconds: float = Field(default=0.05, gt=0)
    search_vector_visibility_poll_max_seconds: float = Field(default=0.5, gt=0)
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
            "http://localhost:3101",
            "http://127.0.0.1:3101",
            "https://capsule-search-workspace.ranhaoxu212.chatgpt.site",
        ]
    )

    @field_validator("ark_api_key", "deepseek_api_key", "milvus_token", mode="before")
    @classmethod
    def empty_secret_is_unset(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def validate_document_chunk_hierarchy(self) -> "Settings":
        if not (
            self.document_chunk_min_tokens
            <= self.document_chunk_target_tokens
            <= self.document_chunk_max_tokens
            <= self.document_chunk_merge_max_tokens
            <= self.document_parent_max_tokens
        ):
            raise ValueError(
                "document token limits must satisfy min <= target <= max <= merge_max <= parent"
            )
        return self

    @model_validator(mode="after")
    def validate_milvus_collections(self) -> "Settings":
        if self.milvus_collection == self.milvus_fused_search_collection:
            raise ValueError("raw and fused-search Milvus collections must be distinct")
        if (
            self.search_vector_visibility_poll_initial_seconds
            > self.search_vector_visibility_poll_max_seconds
        ):
            raise ValueError(
                "search-vector initial visibility poll interval must not exceed its maximum"
            )
        return self

    @model_validator(mode="after")
    def validate_video_task_health_intervals(self) -> "Settings":
        if self.video_task_heartbeat_seconds >= self.video_task_lease_seconds:
            raise ValueError("video task heartbeat must be shorter than its worker lease")
        if self.video_task_progress_timeout_seconds >= self.video_task_hard_timeout_seconds:
            raise ValueError("video task progress timeout must be shorter than its hard timeout")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
