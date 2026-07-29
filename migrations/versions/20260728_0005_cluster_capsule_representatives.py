"""persist cluster capsule representatives and generated summaries

Revision ID: 20260728_0005
Revises: 20260728_0004
Create Date: 2026-07-28 22:15:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260728_0005"
down_revision: str | None = "20260728_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "cluster_capsules",
        sa.Column("model_generated_name", sa.String(length=1024), nullable=True),
    )
    op.add_column(
        "cluster_capsules",
        sa.Column("user_override_name", sa.String(length=1024), nullable=True),
    )
    op.add_column(
        "cluster_capsules",
        sa.Column("model_generated_description", sa.Text(), nullable=True),
    )
    op.add_column(
        "cluster_capsules",
        sa.Column("user_override_description", sa.Text(), nullable=True),
    )
    op.add_column(
        "cluster_capsules",
        sa.Column(
            "common_features",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column(
        "cluster_capsules",
        sa.Column("internal_variance", sa.String(length=16), nullable=True),
    )
    op.add_column(
        "cluster_capsules",
        sa.Column("medoid_asset_id", sa.String(length=64), nullable=True),
    )
    op.create_foreign_key(
        "fk_cluster_capsules_medoid_asset",
        "cluster_capsules",
        "assets",
        ["medoid_asset_id"],
        ["asset_id"],
        ondelete="SET NULL",
    )
    op.create_index(
        op.f("ix_cluster_capsules_medoid_asset_id"),
        "cluster_capsules",
        ["medoid_asset_id"],
        unique=False,
    )
    op.execute(
        "UPDATE cluster_capsules "
        "SET model_generated_name = effective_name, "
        "model_generated_description = effective_description"
    )
    op.alter_column(
        "cluster_capsules",
        "model_generated_name",
        existing_type=sa.String(length=1024),
        nullable=False,
    )
    op.alter_column(
        "cluster_capsules",
        "model_generated_description",
        existing_type=sa.Text(),
        nullable=False,
    )

    op.create_table(
        "cluster_representative_assets",
        sa.Column("cluster_representative_asset_id", sa.String(length=64), nullable=False),
        sa.Column("cluster_capsule_id", sa.String(length=64), nullable=False),
        sa.Column("asset_id", sa.String(length=64), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("distance_to_medoid", sa.Float(), nullable=False),
        sa.Column("membership_probability", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(
            ["asset_id"],
            ["assets.asset_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["cluster_capsule_id"],
            ["cluster_capsules.cluster_capsule_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("cluster_representative_asset_id"),
        sa.UniqueConstraint(
            "cluster_capsule_id",
            "asset_id",
            name="uq_cluster_representative_asset",
        ),
        sa.UniqueConstraint(
            "cluster_capsule_id",
            "rank",
            name="uq_cluster_representative_rank",
        ),
    )
    op.create_index(
        op.f("ix_cluster_representative_assets_asset_id"),
        "cluster_representative_assets",
        ["asset_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_cluster_representative_assets_cluster_capsule_id"),
        "cluster_representative_assets",
        ["cluster_capsule_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_cluster_representative_assets_cluster_capsule_id"),
        table_name="cluster_representative_assets",
    )
    op.drop_index(
        op.f("ix_cluster_representative_assets_asset_id"),
        table_name="cluster_representative_assets",
    )
    op.drop_table("cluster_representative_assets")

    op.drop_index(op.f("ix_cluster_capsules_medoid_asset_id"), table_name="cluster_capsules")
    op.drop_constraint(
        "fk_cluster_capsules_medoid_asset",
        "cluster_capsules",
        type_="foreignkey",
    )
    op.drop_column("cluster_capsules", "medoid_asset_id")
    op.drop_column("cluster_capsules", "internal_variance")
    op.drop_column("cluster_capsules", "common_features")
    op.drop_column("cluster_capsules", "user_override_description")
    op.drop_column("cluster_capsules", "model_generated_description")
    op.drop_column("cluster_capsules", "user_override_name")
    op.drop_column("cluster_capsules", "model_generated_name")
