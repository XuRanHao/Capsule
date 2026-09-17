"""Enqueue current structured memories for their first Milvus projection.

Revision ID: 20260917_0031
Revises: 20260917_0030
Create Date: 2026-09-17
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260917_0031"
down_revision: str | None = "20260917_0030"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Backfill derived-index events without assuming a deployment starts empty."""

    op.execute(
        """
        INSERT INTO agent_memory_vector_outbox (event_id, memory_id, memory_version)
        SELECT
            'memvecout_' || md5(memory_id || ':' || version::text),
            memory_id,
            version
        FROM agent_memories
        ON CONFLICT (memory_id, memory_version) DO NOTHING
        """
    )


def downgrade() -> None:
    # Derived synchronization requests may already be in flight; removing them
    # during a schema downgrade could leave stale vectors, so this is intentional.
    return None
