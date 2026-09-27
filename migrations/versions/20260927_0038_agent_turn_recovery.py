"""Persist Agent turn lifecycle records and link tool audits to turns.

Revision ID: 20260927_0038
Revises: 20260918_0037
Create Date: 2026-09-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260927_0038"
down_revision: str | None = "20260918_0037"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_turns",
        sa.Column("turn_id", sa.String(length=128), nullable=False),
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="running", nullable=False),
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
        sa.CheckConstraint(
            "status IN ('running', 'awaiting_confirmation', 'completed', "
            "'cancelled', 'failed', 'interrupted')",
            name="ck_agent_turns_status",
        ),
        sa.ForeignKeyConstraint(["thread_id"], ["agent_threads.thread_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("turn_id"),
        sa.UniqueConstraint("thread_id", "request_id", name="uq_agent_turns_thread_request"),
    )
    op.create_index("ix_agent_turns_thread_id", "agent_turns", ["thread_id"])
    op.create_index("ix_agent_turns_user_id", "agent_turns", ["user_id"])
    op.create_index("ix_agent_turns_workspace_id", "agent_turns", ["workspace_id"])
    op.create_index(
        "ix_agent_turns_thread_status",
        "agent_turns",
        ["thread_id", "status", "created_at"],
    )
    op.add_column(
        "agent_tool_executions",
        sa.Column("turn_id", sa.String(length=128), nullable=True),
    )
    op.create_index(
        "ix_agent_tool_executions_turn_created",
        "agent_tool_executions",
        ["thread_id", "turn_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_agent_tool_executions_turn_created", table_name="agent_tool_executions")
    op.drop_column("agent_tool_executions", "turn_id")
    op.drop_index("ix_agent_turns_thread_status", table_name="agent_turns")
    op.drop_index("ix_agent_turns_workspace_id", table_name="agent_turns")
    op.drop_index("ix_agent_turns_user_id", table_name="agent_turns")
    op.drop_index("ix_agent_turns_thread_id", table_name="agent_turns")
    op.drop_table("agent_turns")
