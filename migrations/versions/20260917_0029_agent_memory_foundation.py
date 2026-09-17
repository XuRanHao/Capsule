"""Add durable Agent conversations, memory profiles, memories, and Outbox.

Revision ID: 20260917_0029
Revises: 20260910_0028
Create Date: 2026-09-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260917_0029"
down_revision: str | None = "20260910_0028"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_threads",
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=255), server_default="新会话", nullable=False),
        sa.Column("status", sa.String(length=32), server_default="active", nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("summary_topic", sa.String(length=128), nullable=True),
        sa.Column("summary_covered_sequence", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_consolidated_sequence", sa.Integer(), server_default="0", nullable=False),
        sa.Column("memory_revision", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("memory_lease_owner", sa.String(length=128), nullable=True),
        sa.Column("memory_lease_expires_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.PrimaryKeyConstraint("thread_id"),
    )
    op.create_index("ix_agent_threads_user_id", "agent_threads", ["user_id"])
    op.create_index("ix_agent_threads_workspace_id", "agent_threads", ["workspace_id"])
    op.create_index(
        "ix_agent_threads_user_workspace_updated",
        "agent_threads",
        ["user_id", "workspace_id", "updated_at"],
    )

    op.create_table(
        "agent_messages",
        sa.Column("message_id", sa.String(length=64), nullable=False),
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("turn_id", sa.String(length=128), nullable=True),
        sa.Column("request_id", sa.String(length=128), nullable=True),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=True),
        sa.Column("content", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("estimated_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["thread_id"], ["agent_threads.thread_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("message_id"),
        sa.UniqueConstraint("thread_id", "sequence", name="uq_agent_messages_thread_sequence"),
        sa.UniqueConstraint(
            "thread_id",
            "role",
            "request_id",
            name="uq_agent_messages_thread_role_request",
        ),
    )
    op.create_index("ix_agent_messages_thread_id", "agent_messages", ["thread_id"])
    op.create_index(
        "ix_agent_messages_thread_sequence",
        "agent_messages",
        ["thread_id", "sequence"],
    )

    op.create_table(
        "workspace_memory_profiles",
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column(
            "active_topics",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("primary_topic", sa.String(length=128), nullable=True),
        sa.Column("revision", sa.Integer(), server_default="0", nullable=False),
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
        sa.PrimaryKeyConstraint("workspace_id"),
    )

    op.create_table(
        "agent_memories",
        sa.Column("memory_id", sa.String(length=64), nullable=False),
        sa.Column("scope", sa.String(length=16), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=True),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("memory_key", sa.String(length=255), nullable=False),
        sa.Column(
            "value",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("display_text", sa.Text(), nullable=False),
        sa.Column(
            "topics",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("initial_confidence", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("decay_rate", sa.Float(), nullable=False),
        sa.Column(
            "last_reinforced_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=32), server_default="active", nullable=False),
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("supersedes_memory_id", sa.String(length=64), nullable=True),
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
        sa.CheckConstraint("scope IN ('workspace', 'global')", name="ck_agent_memories_scope"),
        sa.CheckConstraint(
            "(scope = 'workspace' AND workspace_id IS NOT NULL) "
            "OR (scope = 'global' AND workspace_id IS NULL)",
            name="ck_agent_memories_scope_workspace",
        ),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.workspace_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("memory_id"),
    )
    op.create_index("ix_agent_memories_user_id", "agent_memories", ["user_id"])
    op.create_index("ix_agent_memories_workspace_id", "agent_memories", ["workspace_id"])
    op.create_index(
        "ix_agent_memories_scope_lookup",
        "agent_memories",
        ["scope", "user_id", "workspace_id", "status", "kind", "memory_key"],
    )
    op.execute(
        """
        CREATE INDEX ix_agent_memories_bm25_search
        ON agent_memories
        USING bm25 (memory_id, memory_key, display_text)
        WITH (key_field = 'memory_id')
        """
    )

    op.create_table(
        "agent_memory_sources",
        sa.Column("source_id", sa.String(length=64), nullable=False),
        sa.Column("memory_id", sa.String(length=64), nullable=False),
        sa.Column("thread_id", sa.String(length=128), nullable=True),
        sa.Column("message_id", sa.String(length=64), nullable=True),
        sa.Column("relation", sa.String(length=32), nullable=False),
        sa.Column("confidence_delta", sa.Float(), server_default="0", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["memory_id"], ["agent_memories.memory_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("source_id"),
    )
    op.create_index("ix_agent_memory_sources_memory_id", "agent_memory_sources", ["memory_id"])
    op.create_index("ix_agent_memory_sources_thread_id", "agent_memory_sources", ["thread_id"])
    op.create_index("ix_agent_memory_sources_message_id", "agent_memory_sources", ["message_id"])
    op.create_index(
        "ix_agent_memory_sources_memory_created",
        "agent_memory_sources",
        ["memory_id", "created_at"],
    )

    op.create_table(
        "agent_memory_outbox",
        sa.Column("event_id", sa.String(length=64), nullable=False),
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("through_sequence", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), server_default="pending", nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "available_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(["thread_id"], ["agent_threads.thread_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("event_id"),
        sa.UniqueConstraint(
            "thread_id",
            "through_sequence",
            name="uq_agent_memory_outbox_thread_through_sequence",
        ),
    )
    op.create_index("ix_agent_memory_outbox_thread_id", "agent_memory_outbox", ["thread_id"])
    op.create_index(
        "ix_agent_memory_outbox_pending",
        "agent_memory_outbox",
        ["status", "available_at", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_agent_memory_outbox_pending", table_name="agent_memory_outbox")
    op.drop_index("ix_agent_memory_outbox_thread_id", table_name="agent_memory_outbox")
    op.drop_table("agent_memory_outbox")
    op.drop_index("ix_agent_memory_sources_memory_created", table_name="agent_memory_sources")
    op.drop_index("ix_agent_memory_sources_message_id", table_name="agent_memory_sources")
    op.drop_index("ix_agent_memory_sources_thread_id", table_name="agent_memory_sources")
    op.drop_index("ix_agent_memory_sources_memory_id", table_name="agent_memory_sources")
    op.drop_table("agent_memory_sources")
    op.drop_index("ix_agent_memories_scope_lookup", table_name="agent_memories")
    op.execute("DROP INDEX IF EXISTS ix_agent_memories_bm25_search")
    op.drop_index("ix_agent_memories_workspace_id", table_name="agent_memories")
    op.drop_index("ix_agent_memories_user_id", table_name="agent_memories")
    op.drop_table("agent_memories")
    op.drop_table("workspace_memory_profiles")
    op.drop_index("ix_agent_messages_thread_sequence", table_name="agent_messages")
    op.drop_index("ix_agent_messages_thread_id", table_name="agent_messages")
    op.drop_table("agent_messages")
    op.drop_index("ix_agent_threads_user_workspace_updated", table_name="agent_threads")
    op.drop_index("ix_agent_threads_workspace_id", table_name="agent_threads")
    op.drop_index("ix_agent_threads_user_id", table_name="agent_threads")
    op.drop_table("agent_threads")
