"""persist measured processing stage durations

Revision ID: 20260729_0007
Revises: 20260729_0006
Create Date: 2026-07-29 16:00:00
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260729_0007"
down_revision: str | None = "20260729_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE processing_jobs
        ADD COLUMN IF NOT EXISTS stage_durations_ms JSONB NOT NULL DEFAULT '{}'::jsonb
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE processing_jobs
        DROP COLUMN IF EXISTS stage_durations_ms
        """
    )
