"""cache completed source assetization by content and processing fingerprint

Revision ID: 20260729_0006
Revises: 20260728_0005
Create Date: 2026-07-29 12:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260729_0006"
down_revision: str | None = "20260728_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "source_files",
        sa.Column("processing_fingerprint", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("source_files", "processing_fingerprint")
