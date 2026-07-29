"""add Asset file tree context

Revision ID: 20260728_0004
Revises: 20260728_0003
Create Date: 2026-07-28 21:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260728_0004"
down_revision: str | None = "20260728_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "assets",
        sa.Column("file_tree_context", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.execute(
        "UPDATE assets AS asset SET file_tree_context = source.file_tree_context "
        "FROM source_files AS source WHERE asset.source_file_id = source.source_file_id"
    )
    op.alter_column(
        "assets",
        "file_tree_context",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        nullable=False,
    )


def downgrade() -> None:
    op.drop_column("assets", "file_tree_context")
