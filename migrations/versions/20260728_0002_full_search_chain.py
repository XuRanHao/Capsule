"""full search chain

Revision ID: 20260728_0002
Revises: 20260728_0001
Create Date: 2026-07-28 17:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260728_0002"
down_revision: str | None = "20260728_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for table in ("source_files", "assets", "embedding_records"):
        op.add_column(
            table,
            sa.Column(
                "project_id",
                sa.String(length=64),
                server_default="project_default",
                nullable=False,
            ),
        )
        op.create_index(op.f(f"ix_{table}_project_id"), table, ["project_id"], unique=False)

    op.create_table(
        "user_favorites",
        sa.Column("favorite_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("created_by", sa.String(length=128), nullable=False),
        sa.Column("asset_id", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["asset_id"], ["assets.asset_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.workspace_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("favorite_id"),
        sa.UniqueConstraint(
            "workspace_id",
            "created_by",
            "asset_id",
            name="uq_user_favorite_workspace_user_asset",
        ),
    )
    op.create_index(op.f("ix_user_favorites_asset_id"), "user_favorites", ["asset_id"])
    op.create_index(op.f("ix_user_favorites_created_by"), "user_favorites", ["created_by"])
    op.create_index(op.f("ix_user_favorites_workspace_id"), "user_favorites", ["workspace_id"])

    op.create_table(
        "search_capsules",
        sa.Column("capsule_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("created_by", sa.String(length=128), nullable=False),
        sa.Column("query_type", sa.String(length=32), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=True),
        sa.Column("query_image_uri", sa.Text(), nullable=True),
        sa.Column("parsed_query", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("filters", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("fusion_method", sa.String(length=64), nullable=False),
        sa.Column("rerank_method", sa.String(length=64), nullable=False),
        sa.Column("search_engine_version", sa.String(length=128), nullable=False),
        sa.Column("embedding_model", sa.String(length=255), nullable=False),
        sa.Column("is_favorite", sa.Boolean(), nullable=False),
        sa.Column(
            "last_used_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.workspace_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("capsule_id"),
    )
    op.create_index(op.f("ix_search_capsules_workspace_id"), "search_capsules", ["workspace_id"])
    op.create_index(op.f("ix_search_capsules_created_by"), "search_capsules", ["created_by"])
    op.create_index(op.f("ix_search_capsules_is_favorite"), "search_capsules", ["is_favorite"])
    op.create_index(op.f("ix_search_capsules_last_used_at"), "search_capsules", ["last_used_at"])

    op.create_table(
        "search_executions",
        sa.Column("execution_id", sa.String(length=64), nullable=False),
        sa.Column("capsule_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("request_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("parsed_query", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("degraded", sa.Boolean(), nullable=False),
        sa.Column("degraded_reasons", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["capsule_id"], ["search_capsules.capsule_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.workspace_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("execution_id"),
    )
    op.create_index(op.f("ix_search_executions_capsule_id"), "search_executions", ["capsule_id"])
    op.create_index(
        op.f("ix_search_executions_workspace_id"), "search_executions", ["workspace_id"]
    )
    op.create_index(op.f("ix_search_executions_status"), "search_executions", ["status"])
    op.create_index(op.f("ix_search_executions_created_at"), "search_executions", ["created_at"])

    op.create_table(
        "search_result_snapshots",
        sa.Column("snapshot_id", sa.String(length=64), nullable=False),
        sa.Column("execution_id", sa.String(length=64), nullable=False),
        sa.Column("capsule_id", sa.String(length=64), nullable=False),
        sa.Column("asset_id", sa.String(length=64), nullable=True),
        sa.Column("result_rank", sa.Integer(), nullable=False),
        sa.Column("final_score", sa.Float(), nullable=False),
        sa.Column("component_scores", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("result_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["asset_id"], ["assets.asset_id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["capsule_id"], ["search_capsules.capsule_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["execution_id"], ["search_executions.execution_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("snapshot_id"),
        sa.UniqueConstraint(
            "execution_id",
            "result_rank",
            name="uq_search_snapshot_execution_rank",
        ),
    )
    op.create_index(
        op.f("ix_search_result_snapshots_execution_id"),
        "search_result_snapshots",
        ["execution_id"],
    )
    op.create_index(
        op.f("ix_search_result_snapshots_capsule_id"),
        "search_result_snapshots",
        ["capsule_id"],
    )
    op.create_index(
        op.f("ix_search_result_snapshots_asset_id"),
        "search_result_snapshots",
        ["asset_id"],
    )
    op.create_index(
        op.f("ix_search_result_snapshots_created_at"),
        "search_result_snapshots",
        ["created_at"],
    )

    op.create_table(
        "query_image_uploads",
        sa.Column("upload_id", sa.String(length=128), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("object_key", sa.Text(), nullable=False),
        sa.Column("content_type", sa.String(length=255), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.workspace_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("upload_id"),
        sa.UniqueConstraint("object_key"),
    )
    op.create_index(
        op.f("ix_query_image_uploads_workspace_id"),
        "query_image_uploads",
        ["workspace_id"],
    )
    op.create_index(
        op.f("ix_query_image_uploads_created_at"),
        "query_image_uploads",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_query_image_uploads_created_at"), table_name="query_image_uploads")
    op.drop_index(op.f("ix_query_image_uploads_workspace_id"), table_name="query_image_uploads")
    op.drop_table("query_image_uploads")
    op.drop_index(
        op.f("ix_search_result_snapshots_created_at"),
        table_name="search_result_snapshots",
    )
    op.drop_index(
        op.f("ix_search_result_snapshots_asset_id"),
        table_name="search_result_snapshots",
    )
    op.drop_index(
        op.f("ix_search_result_snapshots_capsule_id"),
        table_name="search_result_snapshots",
    )
    op.drop_index(
        op.f("ix_search_result_snapshots_execution_id"),
        table_name="search_result_snapshots",
    )
    op.drop_table("search_result_snapshots")
    op.drop_index(op.f("ix_search_executions_created_at"), table_name="search_executions")
    op.drop_index(op.f("ix_search_executions_status"), table_name="search_executions")
    op.drop_index(op.f("ix_search_executions_workspace_id"), table_name="search_executions")
    op.drop_index(op.f("ix_search_executions_capsule_id"), table_name="search_executions")
    op.drop_table("search_executions")
    op.drop_index(op.f("ix_search_capsules_last_used_at"), table_name="search_capsules")
    op.drop_index(op.f("ix_search_capsules_is_favorite"), table_name="search_capsules")
    op.drop_index(op.f("ix_search_capsules_created_by"), table_name="search_capsules")
    op.drop_index(op.f("ix_search_capsules_workspace_id"), table_name="search_capsules")
    op.drop_table("search_capsules")
    op.drop_index(op.f("ix_user_favorites_workspace_id"), table_name="user_favorites")
    op.drop_index(op.f("ix_user_favorites_created_by"), table_name="user_favorites")
    op.drop_index(op.f("ix_user_favorites_asset_id"), table_name="user_favorites")
    op.drop_table("user_favorites")
    for table in ("embedding_records", "assets", "source_files"):
        op.drop_index(op.f(f"ix_{table}_project_id"), table_name=table)
        op.drop_column(table, "project_id")
