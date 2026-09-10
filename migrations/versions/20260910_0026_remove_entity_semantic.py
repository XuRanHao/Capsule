"""Remove the redundant logical entity semantic field.

Revision ID: 20260910_0026
Revises: 20260910_0025
Create Date: 2026-09-10
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260910_0026"
down_revision: str | None = "20260910_0025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("logical_entities", "semantic")


def downgrade() -> None:
    raise RuntimeError("Entity semantic field removal is irreversible.")
