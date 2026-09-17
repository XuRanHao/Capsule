"""Add soft-deletion metadata and lifecycle lookup support for conversations."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260917_0033"
down_revision: str | None = "20260917_0032"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "agent_threads",
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        "ck_agent_threads_status",
        "agent_threads",
        "status IN ('active', 'archived', 'deleted')",
    )
    op.create_index(
        "ix_agent_threads_user_workspace_status_updated",
        "agent_threads",
        ["user_id", "workspace_id", "status", "updated_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_threads_user_workspace_status_updated",
        table_name="agent_threads",
    )
    op.drop_constraint("ck_agent_threads_status", "agent_threads", type_="check")
    op.drop_column("agent_threads", "deleted_at")
