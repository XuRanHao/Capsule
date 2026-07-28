from enum import StrEnum


class AssetType(StrEnum):
    MARKDOWN_BLOCK = "markdown_block"
    IMAGE = "image"
    VIDEO_SEGMENT = "video_segment"


class ProcessingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL_FAILED = "partial_failed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EmbeddingType(StrEnum):
    NATIVE_MULTIMODAL = "native_multimodal"
    ASSET_DESCRIPTION = "asset_description"
    SUBJECT_CONTENT = "subject_content"
    VISUAL_STYLE = "visual_style"


class EmbeddingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class EmbeddingSourceMode(StrEnum):
    ORIGINAL_TEXT = "original_text"
    ORIGINAL_IMAGE = "original_image"
    FEATURE_TEXT = "feature_text"
    DESCRIPTION_TEXT = "description_text"
    DESCRIPTION_FALLBACK = "description_fallback"


class FeatureStatus(StrEnum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    METADATA = "metadata"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class ClusterRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    INSUFFICIENT_DATA = "insufficient_data"
    FAILED = "failed"
