"""Create project-scoped narrative graph storage.

Revision ID: 20260910_0024
Revises: 20260910_0023
Create Date: 2026-09-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260910_0024"
down_revision: str | None = "20260910_0023"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The old automatic material-relationship scaffold cannot express
    # graph-local Entity identifiers or reusable Assets across stories.
    op.execute("DROP TABLE IF EXISTS asset_entity_relations")
    op.execute("DROP TABLE IF EXISTS entity_entity_relations")
    op.execute("DROP TABLE IF EXISTS relation_entity_sources")
    op.execute("DROP TABLE IF EXISTS relation_entities")
    op.create_unique_constraint(
        "uq_assets_asset_workspace",
        "assets",
        ["asset_id", "workspace_id"],
    )

    op.create_table(
        "narrative_graphs",
        sa.Column("graph_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=1024), nullable=False),
        sa.Column("description", sa.Text(), server_default="", nullable=False),
        sa.Column(
            "narrative_context",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
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
        sa.PrimaryKeyConstraint("graph_id"),
        sa.UniqueConstraint("graph_id", "workspace_id", name="uq_narrative_graph_workspace"),
    )
    op.create_index("ix_narrative_graphs_workspace_id", "narrative_graphs", ["workspace_id"])

    op.create_table(
        "graph_assets",
        sa.Column("graph_id", sa.String(length=64), nullable=False),
        sa.Column("asset_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["graph_id", "workspace_id"],
            ["narrative_graphs.graph_id", "narrative_graphs.workspace_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["asset_id", "workspace_id"],
            ["assets.asset_id", "assets.workspace_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("graph_id", "asset_id"),
    )
    op.create_index("ix_graph_assets_asset_id", "graph_assets", ["asset_id"])

    op.create_table(
        "logical_entities",
        sa.Column("graph_id", sa.String(length=64), nullable=False),
        sa.Column("entity_id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=1024), nullable=False),
        sa.Column("entity_type", sa.String(length=128), server_default="", nullable=False),
        sa.Column("semantic", sa.Text(), server_default="", nullable=False),
        sa.Column("description", sa.Text(), server_default="", nullable=False),
        sa.Column(
            "embedding_vector",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("embedding_model", sa.String(length=255), server_default="", nullable=False),
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
        sa.ForeignKeyConstraint(
            ["graph_id"], ["narrative_graphs.graph_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("graph_id", "entity_id"),
    )

    op.create_table(
        "graph_asset_bindings",
        sa.Column("graph_id", sa.String(length=64), nullable=False),
        sa.Column("asset_id", sa.String(length=64), nullable=False),
        sa.Column("entity_id", sa.String(length=64), nullable=False),
        sa.Column(
            "binding_role",
            sa.String(length=128),
            server_default="reference",
            nullable=False,
        ),
        sa.Column("description", sa.Text(), server_default="", nullable=False),
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
        sa.ForeignKeyConstraint(
            ["graph_id", "asset_id"],
            ["graph_assets.graph_id", "graph_assets.asset_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["graph_id", "entity_id"],
            ["logical_entities.graph_id", "logical_entities.entity_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("graph_id", "asset_id", "entity_id"),
    )
    op.create_index(
        "ix_graph_asset_bindings_entity_id",
        "graph_asset_bindings",
        ["graph_id", "entity_id"],
    )

    op.create_table(
        "logical_entity_relations",
        sa.Column("relation_id", sa.String(length=64), nullable=False),
        sa.Column("graph_id", sa.String(length=64), nullable=False),
        sa.Column("source_entity_id", sa.String(length=64), nullable=False),
        sa.Column("target_entity_id", sa.String(length=64), nullable=False),
        sa.Column("relation_type", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), server_default="", nullable=False),
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
        sa.ForeignKeyConstraint(
            ["graph_id", "source_entity_id"],
            ["logical_entities.graph_id", "logical_entities.entity_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["graph_id", "target_entity_id"],
            ["logical_entities.graph_id", "logical_entities.entity_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("relation_id"),
        sa.UniqueConstraint(
            "graph_id",
            "source_entity_id",
            "target_entity_id",
            "relation_type",
            name="uq_logical_entity_relation",
        ),
    )
    op.create_index(
        "ix_logical_entity_relations_graph_source",
        "logical_entity_relations",
        ["graph_id", "source_entity_id"],
    )
    op.create_index(
        "ix_logical_entity_relations_graph_target",
        "logical_entity_relations",
        ["graph_id", "target_entity_id"],
    )


def downgrade() -> None:
    raise RuntimeError(
        "Narrative graph migration replaces retired relation tables and is irreversible."
    )
