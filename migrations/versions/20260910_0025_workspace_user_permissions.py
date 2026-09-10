"""Add workspace membership permissions for Agent tools.

Revision ID: 20260910_0025
Revises: 20260910_0024
Create Date: 2026-09-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260910_0025"
down_revision: str | None = "20260910_0024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "workspace_users",
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column(
            "permission_level",
            sa.String(length=32),
            server_default="read",
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
        sa.CheckConstraint(
            "permission_level IN ('read', 'write', 'destructive', 'admin')",
            name="ck_workspace_users_permission_level",
        ),
        sa.PrimaryKeyConstraint("workspace_id", "user_id"),
    )
    op.create_index("ix_workspace_users_user_id", "workspace_users", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_workspace_users_user_id", table_name="workspace_users")
    op.drop_table("workspace_users")
