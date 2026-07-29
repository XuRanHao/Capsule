from enum import StrEnum


class AssetType(StrEnum):
    MARKDOWN_BLOCK = "markdown_block"
    TEXT_BLOCK = "text_block"
    IMAGE = "image"
    VIDEO_SEGMENT = "video_segment"


class ProcessingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL_FAILED = "partial_failed"
    FAILED = "failed"
    SKIPPED = "skipped"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    RETRYING = "retrying"
    COMPLETED = "completed"
    PARTIAL_FAILED = "partial_failed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineStage(StrEnum):
    DISCOVERING = "discovering"
    PARSING = "parsing"
    SEGMENTING = "segmenting"
    ASSET_STORED = "asset_stored"
    UNDERSTANDING = "understanding"
    FEATURE_READY = "feature_ready"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class AssetNameSource(StrEnum):
    MODEL = "model"
    USER = "user"


class EmbeddingType(StrEnum):
    NATIVE_MULTIMODAL = "native_multimodal"
    ASSET_DESCRIPTION = "asset_description"
    SUBJECT_CONTENT = "subject_content"
    SCENE_THEME = "scene_theme"
    VISUAL_STYLE = "visual_style"
    COLOR_COMPOSITION = "color_composition"
    MOOD_ATMOSPHERE = "mood_atmosphere"
    CHARACTER_STATE_OR_PSYCHOLOGY = "character_state_or_psychology"
    ASSET_USAGE = "asset_usage"
    TARGET_AUDIENCE = "target_audience"
    PROVENANCE = "provenance"
    RIGHTS_VERSION_AUTHORSHIP = "rights_version_authorship"


class EmbeddingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class EmbeddingSourceMode(StrEnum):
    ORIGINAL_TEXT = "original_text"
    ORIGINAL_IMAGE = "original_image"
    ORIGINAL_VIDEO = "original_video"
    FEATURE_TEXT = "feature_text"
    DESCRIPTION_TEXT = "description_text"
    DESCRIPTION_FALLBACK = "description_fallback"


class FeatureStatus(StrEnum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    METADATA = "metadata"
    USER_SUPPLIED = "user_supplied"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class ClusterRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    INSUFFICIENT_DATA = "insufficient_data"
    FAILED = "failed"


class ClusterRepresentativeRole(StrEnum):
    """How an Asset represents a Cluster Capsule."""

    MEDOID = "medoid"
    CORE = "core"
    EDGE = "edge"


class ClusterInternalVariance(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
