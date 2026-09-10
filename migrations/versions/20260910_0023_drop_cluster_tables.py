"""Remove retired material-clustering storage.

Revision ID: 20260910_0023
Revises: 20260908_0022
Create Date: 2026-09-10
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260910_0023"
down_revision: str | None = "20260908_0022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # These tables only checkpointed the retired automatic hierarchy workflow.
    op.execute("DROP TABLE IF EXISTS relation_asset_states")
    op.execute("DROP TABLE IF EXISTS relation_hierarchy_batch_commits")
    op.execute("DROP TABLE IF EXISTS relation_graph_builds")

    # Drop children before their referenced Cluster tables so this also works
    # on databases that enforce every historical foreign-key constraint.
    op.execute("DROP TABLE IF EXISTS cluster_exclusions")
    op.execute("DROP TABLE IF EXISTS cluster_members")
    op.execute("DROP TABLE IF EXISTS cluster_representative_assets")
    op.execute("DROP TABLE IF EXISTS cluster_memberships")
    op.execute("DROP TABLE IF EXISTS clusters")
    op.execute("DROP TABLE IF EXISTS cluster_capsules")
    op.execute("DROP TABLE IF EXISTS cluster_runs")


def downgrade() -> None:
    raise RuntimeError("The retired Cluster tables cannot be restored by downgrade.")
