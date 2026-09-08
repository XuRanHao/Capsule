"""Add idempotent hierarchy workflow commits and source-level vectors.

Revision ID: 20260814_0020
Revises: 20260813_0019
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260814_0020"
down_revision: str | None = "20260813_0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.add_column(
        "relation_entity_sources",
        sa.Column(
            "embedding_vector",
            json_type,
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
    )
    op.add_column(
        "relation_entity_sources",
        sa.Column(
            "embedding_model",
            sa.String(length=255),
            server_default="",
            nullable=False,
        ),
    )
    op.create_table(
        "relation_hierarchy_batch_commits",
        sa.Column("commit_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("operation_id", sa.String(length=255), nullable=False),
        sa.Column("request_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("build_version", sa.Integer(), nullable=False),
        sa.Column("input_revision", sa.String(length=64), nullable=False),
        sa.Column(
            "result_payload",
            json_type,
            server_default=sa.text("'{}'"),
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
        sa.ForeignKeyConstraint(
            ["workspace_id"], ["workspaces.workspace_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("commit_id"),
        sa.UniqueConstraint(
            "workspace_id",
            "operation_id",
            name="uq_relation_hierarchy_batch_commit_operation",
        ),
    )
    op.create_index(
        "ix_relation_hierarchy_batch_commits_workspace_id",
        "relation_hierarchy_batch_commits",
        ["workspace_id"],
    )
    op.create_index(
        "ix_relation_hierarchy_batch_commits_build_version",
        "relation_hierarchy_batch_commits",
        ["build_version"],
    )


def downgrade() -> None:
    op.drop_index("ix_relation_hierarchy_batch_commits_build_version")
    op.drop_index("ix_relation_hierarchy_batch_commits_workspace_id")
    op.drop_table("relation_hierarchy_batch_commits")
    op.drop_column("relation_entity_sources", "embedding_model")
    op.drop_column("relation_entity_sources", "embedding_vector")
