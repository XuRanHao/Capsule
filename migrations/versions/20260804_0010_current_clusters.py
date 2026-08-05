"""Add current clusters, members, and assignment exclusions.

Revision ID: 20260804_0010
Revises: 20260803_0009
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260804_0010"
down_revision: str | None = "20260803_0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "clusters",
        sa.Column("cluster_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("embedding_type", sa.String(length=64), nullable=False),
        sa.Column("mode", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=1024), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("representative_asset_id", sa.String(length=64), nullable=True),
        sa.Column("source_run_id", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "mode IN ('dynamic', 'resident_open', 'resident_manual')",
            name="ck_clusters_mode",
        ),
        sa.ForeignKeyConstraint(
            ["representative_asset_id"],
            ["assets.asset_id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["source_run_id"],
            ["cluster_runs.cluster_run_id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.workspace_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("cluster_id"),
    )
    op.create_index("ix_clusters_embedding_type", "clusters", ["embedding_type"])
    op.create_index("ix_clusters_representative_asset_id", "clusters", ["representative_asset_id"])
    op.create_index("ix_clusters_source_run_id", "clusters", ["source_run_id"])
    op.create_index("ix_clusters_workspace_id", "clusters", ["workspace_id"])
    op.create_index(
        "ix_clusters_workspace_embedding_mode",
        "clusters",
        ["workspace_id", "embedding_type", "mode"],
    )

    op.create_table(
        "cluster_members",
        sa.Column("cluster_id", sa.String(length=64), nullable=False),
        sa.Column("asset_id", sa.String(length=64), nullable=False),
        sa.Column("embedding_type", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "source IN ('full_cluster', 'incremental', 'user')",
            name="ck_cluster_members_source",
        ),
        sa.ForeignKeyConstraint(["asset_id"], ["assets.asset_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["cluster_id"], ["clusters.cluster_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("cluster_id", "asset_id"),
        sa.UniqueConstraint(
            "asset_id",
            "embedding_type",
            name="uq_cluster_member_asset_embedding",
        ),
    )
    op.create_index("ix_cluster_members_asset_id", "cluster_members", ["asset_id"])
    op.create_index("ix_cluster_members_embedding_type", "cluster_members", ["embedding_type"])

    op.create_table(
        "cluster_exclusions",
        sa.Column("cluster_id", sa.String(length=64), nullable=False),
        sa.Column("asset_id", sa.String(length=64), nullable=False),
        sa.Column("created_by", sa.String(length=128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["asset_id"], ["assets.asset_id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["cluster_id"], ["clusters.cluster_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("cluster_id", "asset_id"),
    )
    op.create_index("ix_cluster_exclusions_asset_id", "cluster_exclusions", ["asset_id"])

    # Publish each dimension's latest completed historical run as its initial
    # current state. Reusing the Capsule ID makes the backfill deterministic.
    op.execute(
        """
        WITH ranked_runs AS (
            SELECT cluster_run_id,
                   row_number() OVER (
                       PARTITION BY workspace_id, embedding_type
                       ORDER BY completed_at DESC NULLS LAST,
                                started_at DESC NULLS LAST,
                                cluster_run_id DESC
                   ) AS run_rank
            FROM cluster_runs
            WHERE status = 'completed'
        )
        INSERT INTO clusters (
            cluster_id,
            workspace_id,
            embedding_type,
            mode,
            name,
            description,
            representative_asset_id,
            source_run_id,
            created_at,
            updated_at
        )
        SELECT capsule.cluster_capsule_id,
               capsule.workspace_id,
               capsule.embedding_type,
               'dynamic',
               capsule.effective_name,
               capsule.effective_description,
               capsule.medoid_asset_id,
               capsule.cluster_run_id,
               capsule.created_at,
               capsule.updated_at
        FROM cluster_capsules AS capsule
        JOIN ranked_runs AS run
          ON run.cluster_run_id = capsule.cluster_run_id
         AND run.run_rank = 1
        ON CONFLICT (cluster_id) DO NOTHING
        """
    )
    op.execute(
        """
        INSERT INTO cluster_members (
            cluster_id,
            asset_id,
            embedding_type,
            source,
            score,
            created_at
        )
        SELECT cluster.cluster_id,
               membership.asset_id,
               cluster.embedding_type,
               'full_cluster',
               membership.membership_probability,
               now()
        FROM clusters AS cluster
        JOIN cluster_memberships AS membership
          ON membership.cluster_capsule_id = cluster.cluster_id
         AND membership.is_noise = false
        ON CONFLICT (asset_id, embedding_type) DO NOTHING
        """
    )


def downgrade() -> None:
    op.drop_index("ix_cluster_exclusions_asset_id", table_name="cluster_exclusions")
    op.drop_table("cluster_exclusions")
    op.drop_index("ix_cluster_members_embedding_type", table_name="cluster_members")
    op.drop_index("ix_cluster_members_asset_id", table_name="cluster_members")
    op.drop_table("cluster_members")
    op.drop_index("ix_clusters_workspace_embedding_mode", table_name="clusters")
    op.drop_index("ix_clusters_workspace_id", table_name="clusters")
    op.drop_index("ix_clusters_source_run_id", table_name="clusters")
    op.drop_index("ix_clusters_representative_asset_id", table_name="clusters")
    op.drop_index("ix_clusters_embedding_type", table_name="clusters")
    op.drop_table("clusters")
