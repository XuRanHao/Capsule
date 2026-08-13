"""Distinguish hierarchy and semantic Entity relations.

Revision ID: 20260813_0019
Revises: 20260813_0018
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260813_0019"
down_revision: str | None = "20260813_0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "entity_entity_relations",
        sa.Column(
            "edge_type",
            sa.String(length=32),
            server_default="hierarchy",
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("entity_entity_relations", "edge_type")
