"""Add generation guards for incremental video Asset persistence.

Revision ID: 20260730_0008
Revises: 20260729_0007
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260730_0008"
down_revision: str | None = "20260729_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE source_files ADD COLUMN IF NOT EXISTS "
        "processing_generation INTEGER NOT NULL DEFAULT 0"
    )
    op.execute("ALTER TABLE assets ADD COLUMN IF NOT EXISTS generation INTEGER NOT NULL DEFAULT 0")
    op.execute("CREATE INDEX IF NOT EXISTS ix_assets_generation ON assets (generation)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_assets_generation")
    op.execute("ALTER TABLE assets DROP COLUMN IF EXISTS generation")
    op.execute("ALTER TABLE source_files DROP COLUMN IF EXISTS processing_generation")
