"""Add Agent tool idempotency keys and worker leases.

Revision ID: 20260910_0028
Revises: 20260910_0027
Create Date: 2026-09-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260910_0028"
down_revision: str | None = "20260910_0027"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "agent_tool_executions",
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "agent_tool_executions",
        sa.Column("arguments_hash", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "agent_tool_executions",
        sa.Column("lease_owner", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "agent_tool_executions",
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        """
        UPDATE agent_tool_executions
        SET idempotency_key = operation_id,
            arguments_hash = md5(arguments::text)
        WHERE idempotency_key IS NULL OR arguments_hash IS NULL
        """
    )
    op.alter_column("agent_tool_executions", "idempotency_key", nullable=False)
    op.alter_column("agent_tool_executions", "arguments_hash", nullable=False)
    op.create_unique_constraint(
        "uq_agent_tool_executions_session_idempotency",
        "agent_tool_executions",
        ["user_id", "workspace_id", "thread_id", "idempotency_key"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_agent_tool_executions_session_idempotency",
        "agent_tool_executions",
        type_="unique",
    )
    op.drop_column("agent_tool_executions", "lease_expires_at")
    op.drop_column("agent_tool_executions", "lease_owner")
    op.drop_column("agent_tool_executions", "arguments_hash")
    op.drop_column("agent_tool_executions", "idempotency_key")
