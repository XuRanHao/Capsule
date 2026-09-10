from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
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


class WorkspaceUser(Base, TimestampMixin):
    """User membership and graph permissions inside one Workspace."""

    __tablename__ = "workspace_users"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    permission_level: Mapped[str] = mapped_column(
        String(32),
        default="read",
        nullable=False,
    )


class AgentToolExecution(Base, TimestampMixin):
    """Durable audit record for one Agent tool operation."""

    __tablename__ = "agent_tool_executions"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "workspace_id",
            "thread_id",
            "idempotency_key",
            name="uq_agent_tool_executions_session_idempotency",
        ),
        Index(
            "ix_agent_tool_executions_session_created",
            "user_id",
            "workspace_id",
            "thread_id",
            "created_at",
        ),
    )

    operation_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=id_factory("op")
    )
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    arguments_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    call_id: Mapped[str] = mapped_column(String(128), nullable=False)
    thread_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    graph_id: Mapped[str | None] = mapped_column(String(64))
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    required_permission: Mapped[str | None] = mapped_column(String(64))
    arguments: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    confirmation_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="not_required"
    )
    execution_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="created"
    )
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output: Mapped[Any | None] = mapped_column(JSONB)
    error_code: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SourceFile(Base, TimestampMixin):
    __tablename__ = "source_files"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "relative_path",
            name="uq_source_file_workspace_relative_path",
        ),
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
    processing_fingerprint: Mapped[str | None] = mapped_column(String(64))
    processing_generation: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
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
    __table_args__ = (
        UniqueConstraint("source_file_id", "asset_key", name="uq_asset_source_key"),
        UniqueConstraint("asset_id", "workspace_id", name="uq_assets_asset_workspace"),
        UniqueConstraint(
            "parent_asset_id",
            "child_order",
            name="uq_asset_parent_child_order",
            deferrable=True,
            initially="DEFERRED",
        ),
        CheckConstraint(
            "(index_role IN ('standalone', 'parent') AND parent_asset_id IS NULL "
            "AND child_order IS NULL) OR "
            "(index_role = 'child' AND parent_asset_id IS NOT NULL "
            "AND child_order IS NOT NULL AND child_order >= 0)",
            name="ck_assets_index_hierarchy",
        ),
    )

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
    file_type: Mapped[str] = mapped_column(String(32), nullable=False)
    asset_key: Mapped[str] = mapped_column(String(64), nullable=False)
    index_role: Mapped[str] = mapped_column(
        String(16),
        default="standalone",
        nullable=False,
        index=True,
    )
    parent_asset_id: Mapped[str | None] = mapped_column(
        ForeignKey("assets.asset_id", ondelete="CASCADE"),
        index=True,
    )
    child_order: Mapped[int | None] = mapped_column(Integer)
    generation: Mapped[int] = mapped_column(Integer, default=0, nullable=False, index=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    asset_name: Mapped[str | None] = mapped_column(String(1024))
    asset_name_source: Mapped[str | None] = mapped_column(String(32))
    asset_description: Mapped[str | None] = mapped_column(Text)
    asset_features: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    file_tree_context: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    source_contexts: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )
    file_info: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    source_locator: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    raw_content: Mapped[str | None] = mapped_column(Text)
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


class ProcessingJob(Base, TimestampMixin):
    __tablename__ = "processing_jobs"

    job_id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=id_factory("job"),
    )
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        index=True,
    )
    job_type: Mapped[str] = mapped_column(String(64), default="assetization", nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="queued", nullable=False, index=True)
    current_stage: Mapped[str] = mapped_column(
        String(32), default="discovering", nullable=False, index=True
    )
    input_path: Mapped[str] = mapped_column(Text, nullable=False)
    total_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_info: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    stage_durations_ms: Mapped[dict[str, float]] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )
    post_asset_action: Mapped[str] = mapped_column(String(32), default="none", nullable=False)
    dispatch_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    assetization_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    workflow_owner_id: Mapped[str | None] = mapped_column(String(255), index=True)
    workflow_lease_token: Mapped[str | None] = mapped_column(String(64))
    workflow_lease_deadline_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


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


class NarrativeGraph(Base, TimestampMixin):
    """One user-owned narrative graph within a workspace."""

    __tablename__ = "narrative_graphs"
    __table_args__ = (
        UniqueConstraint("graph_id", "workspace_id", name="uq_narrative_graph_workspace"),
    )

    graph_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=id_factory("graph")
    )
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.workspace_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(1024), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    narrative_context: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)


class GraphAsset(Base):
    """One Asset selected into a narrative graph."""

    __tablename__ = "graph_assets"

    __table_args__ = (
        ForeignKeyConstraint(
            ["graph_id", "workspace_id"],
            ["narrative_graphs.graph_id", "narrative_graphs.workspace_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["asset_id", "workspace_id"],
            ["assets.asset_id", "assets.workspace_id"],
            ondelete="CASCADE",
        ),
    )

    graph_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class LogicalEntity(Base, TimestampMixin):
    """A narrative concept such as a character, event, place, or scene."""

    __tablename__ = "logical_entities"

    graph_id: Mapped[str] = mapped_column(
        ForeignKey("narrative_graphs.graph_id", ondelete="CASCADE"), primary_key=True
    )
    entity_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(1024), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")


class GraphAssetBinding(Base, TimestampMixin):
    """A many-to-many binding between selected Assets and logical Entities."""

    __tablename__ = "graph_asset_bindings"
    __table_args__ = (
        ForeignKeyConstraint(
            ["graph_id", "asset_id"],
            ["graph_assets.graph_id", "graph_assets.asset_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["graph_id", "entity_id"],
            ["logical_entities.graph_id", "logical_entities.entity_id"],
            ondelete="CASCADE",
        ),
    )

    graph_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    entity_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    binding_role: Mapped[str] = mapped_column(String(128), nullable=False, default="reference")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")


class LogicalEntityRelation(Base, TimestampMixin):
    """A typed directed relation between two Entities in one narrative graph."""

    __tablename__ = "logical_entity_relations"
    __table_args__ = (
        ForeignKeyConstraint(
            ["graph_id", "source_entity_id"],
            ["logical_entities.graph_id", "logical_entities.entity_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["graph_id", "target_entity_id"],
            ["logical_entities.graph_id", "logical_entities.entity_id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "graph_id",
            "source_entity_id",
            "target_entity_id",
            "relation_type",
            name="uq_logical_entity_relation",
        ),
    )

    relation_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=id_factory("graphrel")
    )
    graph_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_entity_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    target_entity_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    relation_type: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")


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
