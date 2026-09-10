"""Add source-level Entity vectors.

Revision ID: 20260814_0020
Revises: 20260813_0019
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260814_0020"
down_revision: str | None = "20260813_0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.add_column(
        "relation_entity_sources",
        sa.Column(
            "embedding_vector",
            json_type,
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
    )
    op.add_column(
        "relation_entity_sources",
        sa.Column(
            "embedding_model",
            sa.String(length=255),
            server_default="",
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("relation_entity_sources", "embedding_model")
    op.drop_column("relation_entity_sources", "embedding_vector")
