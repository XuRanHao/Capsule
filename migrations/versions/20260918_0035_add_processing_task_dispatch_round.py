"""Persist durable processing-task dispatch rounds.

Revision ID: 20260918_0035
Revises: 20260917_0034
Create Date: 2026-09-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260918_0035"
down_revision: str | None = "20260917_0034"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # A server default backfills existing rows while keeping direct SQL inserts
    # safe; the ORM also supplies the same default for normal submissions.
    op.add_column(
        "processing_tasks",
        sa.Column("dispatch_round", sa.Integer(), server_default="1", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("processing_tasks", "dispatch_round")
