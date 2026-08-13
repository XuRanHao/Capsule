"""Persist relationships between virtual Entity nodes.

Revision ID: 20260813_0018
Revises: 20260812_0017
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260813_0018"
down_revision: str | None = "20260812_0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "entity_entity_relations",
        sa.Column("relation_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("source_entity_id", sa.String(length=64), nullable=False),
        sa.Column("target_entity_id", sa.String(length=64), nullable=False),
        sa.Column("relation", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), server_default="", nullable=False),
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
        sa.ForeignKeyConstraint(
            ["source_entity_id"],
            ["relation_entities.entity_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["target_entity_id"],
            ["relation_entities.entity_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.workspace_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("relation_id"),
        sa.UniqueConstraint(
            "workspace_id",
            "source_entity_id",
            "target_entity_id",
            name="uq_entity_entity_relation",
        ),
    )
    op.create_index(
        "ix_entity_entity_relations_workspace_id",
        "entity_entity_relations",
        ["workspace_id"],
    )
    op.create_index(
        "ix_entity_entity_relations_source_entity_id",
        "entity_entity_relations",
        ["source_entity_id"],
    )
    op.create_index(
        "ix_entity_entity_relations_target_entity_id",
        "entity_entity_relations",
        ["target_entity_id"],
    )


def downgrade() -> None:
    op.drop_table("entity_entity_relations")
