"""Persist relation Entity title-and-description embeddings.

Revision ID: 20260812_0016
Revises: 20260812_0015
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260812_0016"
down_revision: str | None = "20260812_0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "relation_entities",
        sa.Column(
            "embedding_vector",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column(
        "relation_entities",
        sa.Column(
            "embedding_model",
            sa.String(length=255),
            server_default="",
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("relation_entities", "embedding_model")
    op.drop_column("relation_entities", "embedding_vector")
