"""Add durable Milvus synchronization requests for Agent memories.

Revision ID: 20260917_0030
Revises: 20260917_0029
Create Date: 2026-09-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260917_0030"
down_revision: str | None = "20260917_0029"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "agent_memories",
        sa.Column("vector_lease_owner", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "agent_memories",
        sa.Column("vector_lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "agent_memory_vector_outbox",
        sa.Column("event_id", sa.String(length=64), nullable=False),
        sa.Column("memory_id", sa.String(length=64), nullable=False),
        sa.Column("memory_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="pending", nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "available_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.PrimaryKeyConstraint("event_id"),
        sa.UniqueConstraint(
            "memory_id",
            "memory_version",
            name="uq_agent_memory_vector_outbox_memory_version",
        ),
    )
    op.create_index(
        "ix_agent_memory_vector_outbox_memory_id",
        "agent_memory_vector_outbox",
        ["memory_id"],
    )
    op.create_index(
        "ix_agent_memory_vector_outbox_pending",
        "agent_memory_vector_outbox",
        ["status", "available_at", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_memory_vector_outbox_pending",
        table_name="agent_memory_vector_outbox",
    )
    op.drop_index(
        "ix_agent_memory_vector_outbox_memory_id",
        table_name="agent_memory_vector_outbox",
    )
    op.drop_table("agent_memory_vector_outbox")
    op.drop_column("agent_memories", "vector_lease_expires_at")
    op.drop_column("agent_memories", "vector_lease_owner")
