"""Add durable Agent tool execution audit records.

Revision ID: 20260910_0027
Revises: 20260910_0026
Create Date: 2026-09-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260910_0027"
down_revision: str | None = "20260910_0026"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_tool_executions",
        sa.Column("operation_id", sa.String(length=64), nullable=False),
        sa.Column("call_id", sa.String(length=128), nullable=False),
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("graph_id", sa.String(length=64), nullable=True),
        sa.Column("tool_name", sa.String(length=128), nullable=False),
        sa.Column("required_permission", sa.String(length=64), nullable=True),
        sa.Column(
            "arguments",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "confirmation_status",
            sa.String(length=32),
            server_default="not_required",
            nullable=False,
        ),
        sa.Column(
            "execution_status",
            sa.String(length=32),
            server_default="created",
            nullable=False,
        ),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("output", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.PrimaryKeyConstraint("operation_id"),
    )
    op.create_index("ix_agent_tool_executions_user_id", "agent_tool_executions", ["user_id"])
    op.create_index(
        "ix_agent_tool_executions_workspace_id", "agent_tool_executions", ["workspace_id"]
    )
    op.create_index(
        "ix_agent_tool_executions_thread_id", "agent_tool_executions", ["thread_id"]
    )
    op.create_index(
        "ix_agent_tool_executions_session_created",
        "agent_tool_executions",
        ["user_id", "workspace_id", "thread_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_tool_executions_session_created", table_name="agent_tool_executions"
    )
    op.drop_index("ix_agent_tool_executions_thread_id", table_name="agent_tool_executions")
    op.drop_index(
        "ix_agent_tool_executions_workspace_id", table_name="agent_tool_executions"
    )
    op.drop_index("ix_agent_tool_executions_user_id", table_name="agent_tool_executions")
    op.drop_table("agent_tool_executions")
