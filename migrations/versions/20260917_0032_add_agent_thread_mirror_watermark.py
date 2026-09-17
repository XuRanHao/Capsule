"""Add a durable message watermark for Agent-session hot mirrors.

Revision ID: 20260917_0032
Revises: 20260917_0031
Create Date: 2026-09-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260917_0032"
down_revision: str | None = "20260917_0031"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "agent_threads",
        sa.Column(
            "last_message_sequence",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
    )
    op.execute(
        """
        UPDATE agent_threads AS thread
        SET last_message_sequence = COALESCE(
            (
                SELECT MAX(message.sequence)
                FROM agent_messages AS message
                WHERE message.thread_id = thread.thread_id
            ),
            0
        )
        """
    )


def downgrade() -> None:
    op.drop_column("agent_threads", "last_message_sequence")
