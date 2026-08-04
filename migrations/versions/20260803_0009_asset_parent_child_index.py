"""Add parent/child Asset hierarchy metadata for hierarchical retrieval.

Revision ID: 20260803_0009
Revises: 20260730_0008
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260803_0009"
down_revision: str | None = "20260730_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE assets ADD COLUMN IF NOT EXISTS "
        "index_role VARCHAR(16) NOT NULL DEFAULT 'standalone'"
    )
    op.execute("ALTER TABLE assets ADD COLUMN IF NOT EXISTS parent_asset_id VARCHAR(64)")
    op.execute("ALTER TABLE assets ADD COLUMN IF NOT EXISTS child_order INTEGER")
    op.execute("CREATE INDEX IF NOT EXISTS ix_assets_index_role ON assets (index_role)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_assets_parent_asset_id ON assets (parent_asset_id)")
    op.create_foreign_key(
        "fk_assets_parent_asset_id_assets",
        "assets",
        "assets",
        ["parent_asset_id"],
        ["asset_id"],
        ondelete="CASCADE",
    )
    op.create_unique_constraint(
        "uq_asset_parent_child_order",
        "assets",
        ["parent_asset_id", "child_order"],
        deferrable=True,
        initially="DEFERRED",
    )
    op.create_check_constraint(
        "ck_assets_index_hierarchy",
        "assets",
        "(index_role IN ('standalone', 'parent') AND parent_asset_id IS NULL "
        "AND child_order IS NULL) OR "
        "(index_role = 'child' AND parent_asset_id IS NOT NULL "
        "AND child_order IS NOT NULL AND child_order >= 0)",
    )


def downgrade() -> None:
    op.drop_constraint("ck_assets_index_hierarchy", "assets", type_="check")
    op.drop_constraint("uq_asset_parent_child_order", "assets", type_="unique")
    op.drop_constraint("fk_assets_parent_asset_id_assets", "assets", type_="foreignkey")
    op.execute("DROP INDEX IF EXISTS ix_assets_parent_asset_id")
    op.execute("DROP INDEX IF EXISTS ix_assets_index_role")
    op.execute("ALTER TABLE assets DROP COLUMN IF EXISTS child_order")
    op.execute("ALTER TABLE assets DROP COLUMN IF EXISTS parent_asset_id")
    op.execute("ALTER TABLE assets DROP COLUMN IF EXISTS index_role")
