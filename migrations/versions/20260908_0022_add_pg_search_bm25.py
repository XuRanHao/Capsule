"""Enable pg_search and index Asset text with BM25.

Revision ID: 20260908_0022
Revises: 20260907_0021
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260908_0022"
down_revision: str | None = "20260907_0021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # pg_search declares pgvector as a dependency; CASCADE installs it on a
    # clean database before the BM25 index is created.
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_search CASCADE")
    op.execute(
        """
        CREATE INDEX ix_assets_bm25_search
        ON assets
        USING bm25 (
            asset_id,
            file_name,
            asset_name,
            raw_content,
            asset_description
        )
        WITH (key_field = 'asset_id')
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_assets_bm25_search")
    op.execute("DROP EXTENSION IF EXISTS pg_search")
