"""Add a distributed single-active-invocation lease to Agent conversations."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260917_0034"
down_revision: str | None = "20260917_0033"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "agent_threads",
        sa.Column("turn_lease_owner", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "agent_threads",
        sa.Column("turn_lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent_threads", "turn_lease_expires_at")
    op.drop_column("agent_threads", "turn_lease_owner")
