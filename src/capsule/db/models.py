from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from capsule.db.base import Base, id_factory


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Workspace(Base, TimestampMixin):
    __tablename__ = "workspaces"

    workspace_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=id_factory("workspace"),
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)


class SourceFile(Base, TimestampMixin):
    __tablename__ = "source_files"
    __table_args__ = (
        UniqueConstraint("workspace_id", "sha256", name="uq_source_file_workspace_sha256"),
    )

    source_file_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=id_factory("src"),
    )
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        index=True,
    )
    project_id: Mapped[str] = mapped_column(
        String(64),
        default="project_default",
        nullable=False,
        index=True,
    )
    original_file_name: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_type: Mapped[str] = mapped_column(String(32), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(255), nullable=False)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_tree_context: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    processing_status: Mapped[str] = mapped_column(
        String(32),
        default="pending",
        nullable=False,
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text)


class Asset(Base, TimestampMixin):
    __tablename__ = "assets"

    asset_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=id_factory("asset"),
    )
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        index=True,
    )
    project_id: Mapped[str] = mapped_column(
        String(64),
        default="project_default",
        nullable=False,
        index=True,
    )
    source_file_id: Mapped[str] = mapped_column(
        ForeignKey("source_files.source_file_id", ondelete="CASCADE"),
        index=True,
    )
    asset_type: Mapped[str] = mapped_column(String(32), index=True)
    file_name: Mapped[str] = mapped_column(String(1024), nullable=False)
    asset_name: Mapped[str | None] = mapped_column(String(1024))
    asset_description: Mapped[str | None] = mapped_column(Text)
    asset_features: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    source_contexts: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )
    file_info: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    source_locator: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    raw_content: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    derived_file_uri: Mapped[str | None] = mapped_column(Text)
    preview_uri: Mapped[str | None] = mapped_column(Text)
    processing_status: Mapped[str] = mapped_column(
        String(32),
        default="pending",
        nullable=False,
        index=True,
    )
    feature_revision: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    embedding_revision: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)


class EmbeddingRecord(Base):
    __tablename__ = "embedding_records"
    __table_args__ = (
        UniqueConstraint(
            "asset_id",
            "embedding_type",
            "model_name",
            "dimension",
            "source_content_hash",
            name="uq_embedding_logical_input",
        ),
    )

    embedding_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=id_factory("emb"),
    )
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        index=True,
    )
    project_id: Mapped[str] = mapped_column(
        String(64),
        default="project_default",
        nullable=False,
        index=True,
    )
    asset_id: Mapped[str] = mapped_column(
        ForeignKey("assets.asset_id", ondelete="CASCADE"),
        index=True,
    )
    embedding_type: Mapped[str] = mapped_column(String(64), index=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    source_content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_source_mode: Mapped[str] = mapped_column(String(64), nullable=False)
    milvus_collection: Mapped[str] = mapped_column(String(255), nullable=False)
    milvus_primary_key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False, index=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    usage: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class ModelCallLog(Base):
    __tablename__ = "model_call_logs"
    __table_args__ = (
        UniqueConstraint(
            "asset_id",
            "operation",
            "model_name",
            "input_hash",
            name="uq_model_call_logical_input",
        ),
    )

    call_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=id_factory("call"),
    )
    asset_id: Mapped[str | None] = mapped_column(
        ForeignKey("assets.asset_id", ondelete="CASCADE"),
        index=True,
    )
    operation: Mapped[str] = mapped_column(String(64), nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(128))
    usage: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    raw_response: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class ClusterRun(Base):
    __tablename__ = "cluster_runs"

    cluster_run_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=id_factory("run"),
    )
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        index=True,
    )
    embedding_type: Mapped[str] = mapped_column(String(64), index=True)
    input_embedding_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    preprocessing: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    cluster_count: Mapped[int | None] = mapped_column(Integer)
    noise_count: Mapped[int | None] = mapped_column(Integer)
    noise_ratio: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ClusterCapsule(Base, TimestampMixin):
    __tablename__ = "cluster_capsules"
    __table_args__ = (
        UniqueConstraint(
            "cluster_run_id",
            "cluster_label",
            name="uq_cluster_capsule_run_label",
        ),
    )

    cluster_capsule_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=id_factory("cc"),
    )
    cluster_run_id: Mapped[str] = mapped_column(
        ForeignKey("cluster_runs.cluster_run_id", ondelete="CASCADE"),
        index=True,
    )
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        index=True,
    )
    embedding_type: Mapped[str] = mapped_column(String(64), nullable=False)
    cluster_label: Mapped[int] = mapped_column(Integer, nullable=False)
    effective_name: Mapped[str] = mapped_column(String(1024), nullable=False)
    effective_description: Mapped[str] = mapped_column(Text, nullable=False)
    keywords: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    member_count: Mapped[int] = mapped_column(Integer, nullable=False)
    average_membership_probability: Mapped[float] = mapped_column(Float, nullable=False)
    representative_asset_ids: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )
    is_favorite: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class ClusterMembership(Base):
    __tablename__ = "cluster_memberships"
    __table_args__ = (
        UniqueConstraint(
            "cluster_run_id",
            "asset_id",
            name="uq_cluster_membership_run_asset",
        ),
    )

    membership_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=id_factory("membership"),
    )
    cluster_run_id: Mapped[str] = mapped_column(
        ForeignKey("cluster_runs.cluster_run_id", ondelete="CASCADE"),
        index=True,
    )
    cluster_capsule_id: Mapped[str | None] = mapped_column(
        ForeignKey("cluster_capsules.cluster_capsule_id", ondelete="CASCADE"),
        index=True,
    )
    asset_id: Mapped[str] = mapped_column(
        ForeignKey("assets.asset_id", ondelete="CASCADE"),
        index=True,
    )
    hdbscan_label: Mapped[int] = mapped_column(Integer, nullable=False)
    membership_probability: Mapped[float] = mapped_column(Float, nullable=False)
    is_noise: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    distance_to_representative: Mapped[float | None] = mapped_column(Float)


class UserFavorite(Base):
    __tablename__ = "user_favorites"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "created_by",
            "asset_id",
            name="uq_user_favorite_workspace_user_asset",
        ),
    )

    favorite_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=id_factory("favorite"),
    )
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        index=True,
    )
    created_by: Mapped[str] = mapped_column(String(128), index=True)
    asset_id: Mapped[str] = mapped_column(
        ForeignKey("assets.asset_id", ondelete="CASCADE"),
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class SearchCapsule(Base, TimestampMixin):
    __tablename__ = "search_capsules"

    capsule_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=id_factory("search_capsule"),
    )
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        index=True,
    )
    created_by: Mapped[str] = mapped_column(String(128), index=True)
    query_type: Mapped[str] = mapped_column(String(32), nullable=False)
    query_text: Mapped[str | None] = mapped_column(Text)
    query_image_uri: Mapped[str | None] = mapped_column(Text)
    parsed_query: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    filters: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    fusion_method: Mapped[str] = mapped_column(String(64), nullable=False)
    rerank_method: Mapped[str] = mapped_column(String(64), nullable=False)
    search_engine_version: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    is_favorite: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    last_used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )


class SearchExecution(Base):
    __tablename__ = "search_executions"

    execution_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=id_factory("search_exec"),
    )
    capsule_id: Mapped[str] = mapped_column(
        ForeignKey("search_capsules.capsule_id", ondelete="CASCADE"),
        index=True,
    )
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        index=True,
    )
    request_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    parsed_query: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    degraded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    degraded_reasons: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )


class SearchResultSnapshot(Base):
    __tablename__ = "search_result_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "execution_id",
            "result_rank",
            name="uq_search_snapshot_execution_rank",
        ),
    )

    snapshot_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=id_factory("snapshot"),
    )
    execution_id: Mapped[str] = mapped_column(
        ForeignKey("search_executions.execution_id", ondelete="CASCADE"),
        index=True,
    )
    capsule_id: Mapped[str] = mapped_column(
        ForeignKey("search_capsules.capsule_id", ondelete="CASCADE"),
        index=True,
    )
    asset_id: Mapped[str | None] = mapped_column(
        ForeignKey("assets.asset_id", ondelete="SET NULL"),
        index=True,
    )
    result_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    final_score: Mapped[float] = mapped_column(Float, nullable=False)
    component_scores: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    result_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )


class QueryImageUpload(Base):
    __tablename__ = "query_image_uploads"

    upload_id: Mapped[str] = mapped_column(
        String(128),
        primary_key=True,
        default=id_factory("query_image"),
    )
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        index=True,
    )
    object_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    content_type: Mapped[str] = mapped_column(String(255), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
