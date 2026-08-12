"""Persist per-Asset relationship processing revisions.

Revision ID: 20260812_0015
Revises: 20260812_0014
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260812_0015"
down_revision: str | None = "20260812_0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "relation_asset_states",
        sa.Column("state_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("asset_id", sa.String(length=64), nullable=False),
        sa.Column("asset_revision", sa.String(length=64), nullable=False),
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
            ["asset_id"], ["assets.asset_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"], ["workspaces.workspace_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("state_id"),
        sa.UniqueConstraint(
            "workspace_id", "asset_id", name="uq_relation_asset_state"
        ),
    )
    op.create_index(
        "ix_relation_asset_states_workspace_id",
        "relation_asset_states",
        ["workspace_id"],
    )
    op.create_index(
        "ix_relation_asset_states_asset_id",
        "relation_asset_states",
        ["asset_id"],
    )


def downgrade() -> None:
    op.drop_table("relation_asset_states")
