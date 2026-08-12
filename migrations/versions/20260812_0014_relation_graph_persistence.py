"""Persist relation entities and Asset-to-Entity decisions.

Revision ID: 20260812_0014
Revises: 20260812_0013
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260812_0014"
down_revision: str | None = "20260812_0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "relation_graph_builds",
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("input_revision", sa.String(length=64), nullable=False),
        sa.Column("build_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column(
            "subject_cluster_status",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
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
        sa.PrimaryKeyConstraint("workspace_id"),
    )
    op.create_index(
        "ix_relation_graph_builds_input_revision", "relation_graph_builds", ["input_revision"]
    )

    op.create_table(
        "relation_entities",
        sa.Column("entity_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=1024), nullable=False),
        sa.Column("semantic", sa.Text(), nullable=False),
        sa.Column("origins", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("descriptions", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("candidate_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("merge_reason", sa.Text(), nullable=False),
        sa.Column("build_version", sa.Integer(), nullable=False),
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
        sa.PrimaryKeyConstraint("entity_id"),
    )
    op.create_index("ix_relation_entities_workspace_id", "relation_entities", ["workspace_id"])

    op.create_table(
        "relation_entity_sources",
        sa.Column("entity_id", sa.String(length=64), nullable=False),
        sa.Column("candidate_id", sa.String(length=255), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("origin", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=1024), nullable=False),
        sa.Column("semantic", sa.Text(), nullable=False),
        sa.Column("asset_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(["entity_id"], ["relation_entities.entity_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.workspace_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("entity_id", "candidate_id"),
    )
    op.create_index(
        "ix_relation_entity_sources_workspace_id", "relation_entity_sources", ["workspace_id"]
    )

    op.create_table(
        "asset_entity_relations",
        sa.Column("relation_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("asset_id", sa.String(length=64), nullable=False),
        sa.Column("entity_id", sa.String(length=64), nullable=False),
        sa.Column("establishes_relation", sa.Boolean(), nullable=False),
        sa.Column("relation", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("content_subject", sa.Text(), nullable=False),
        sa.Column("build_version", sa.Integer(), nullable=False),
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
        sa.ForeignKeyConstraint(["asset_id"], ["assets.asset_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["entity_id"], ["relation_entities.entity_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.workspace_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("relation_id"),
        sa.UniqueConstraint(
            "workspace_id", "asset_id", "entity_id", name="uq_asset_entity_relation"
        ),
    )
    op.create_index(
        "ix_asset_entity_relations_workspace_id", "asset_entity_relations", ["workspace_id"]
    )
    op.create_index("ix_asset_entity_relations_asset_id", "asset_entity_relations", ["asset_id"])
    op.create_index("ix_asset_entity_relations_entity_id", "asset_entity_relations", ["entity_id"])


def downgrade() -> None:
    op.drop_table("asset_entity_relations")
    op.drop_table("relation_entity_sources")
    op.drop_table("relation_entities")
    op.drop_table("relation_graph_builds")
