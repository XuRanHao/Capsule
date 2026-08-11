"""Remove Agent rerank persistence.

Revision ID: 20260810_0011
Revises: 20260804_0010
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260810_0011"
down_revision: str | None = "20260804_0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("search_capsules", "rerank_method")


def downgrade() -> None:
    op.add_column(
        "search_capsules",
        sa.Column(
            "rerank_method",
            sa.String(length=64),
            server_default="off",
            nullable=False,
        ),
    )
    op.alter_column("search_capsules", "rerank_method", server_default=None)
