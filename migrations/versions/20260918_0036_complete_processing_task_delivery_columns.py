"""Complete durable processing-task delivery columns.

Some existing development databases received these columns before their
Alembic revision was recorded.  Use PostgreSQL's conditional form so both
those databases and clean installations converge on the model schema.

Revision ID: 20260918_0036
Revises: 20260918_0035
Create Date: 2026-09-18
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260918_0036"
down_revision: str | None = "20260918_0035"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # `IF NOT EXISTS` keeps the revision safe for databases that have the
    # two publication timestamps from a prior untracked schema change.
    op.execute(
        "ALTER TABLE processing_tasks "
        "ADD COLUMN IF NOT EXISTS retry_event_id VARCHAR(160)"
    )
    op.execute(
        "ALTER TABLE processing_tasks "
        "ADD COLUMN IF NOT EXISTS last_published_at TIMESTAMP WITH TIME ZONE"
    )
    op.execute(
        "ALTER TABLE processing_tasks "
        "ADD COLUMN IF NOT EXISTS dlq_published_at TIMESTAMP WITH TIME ZONE"
    )


def downgrade() -> None:
    op.drop_column("processing_tasks", "dlq_published_at")
    op.drop_column("processing_tasks", "last_published_at")
    op.drop_column("processing_tasks", "retry_event_id")
