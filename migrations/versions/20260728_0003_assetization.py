"""add assetization persistence

Revision ID: 20260728_0003
Revises: 20260728_0002
Create Date: 2026-07-28 19:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260728_0003"
down_revision: str | None = "20260728_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("uq_source_file_workspace_sha256", "source_files", type_="unique")
    op.create_unique_constraint(
        "uq_source_file_workspace_relative_path",
        "source_files",
        ["workspace_id", "relative_path"],
    )

    op.add_column("assets", sa.Column("file_type", sa.String(length=32), nullable=True))
    op.add_column("assets", sa.Column("asset_key", sa.String(length=64), nullable=True))
    op.add_column("assets", sa.Column("content_hash", sa.String(length=64), nullable=True))
    op.add_column("assets", sa.Column("asset_name_source", sa.String(length=32), nullable=True))
    op.execute(
        "UPDATE assets AS asset SET file_type = source.file_type "
        "FROM source_files AS source WHERE asset.source_file_id = source.source_file_id"
    )
    op.execute("UPDATE assets SET asset_key = asset_id, content_hash = asset_id")
    op.alter_column("assets", "file_type", nullable=False)
    op.alter_column("assets", "asset_key", nullable=False)
    op.alter_column("assets", "content_hash", nullable=False)
    op.create_unique_constraint("uq_asset_source_key", "assets", ["source_file_id", "asset_key"])
    op.alter_column(
        "assets",
        "raw_content",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        type_=sa.Text(),
        existing_nullable=True,
        postgresql_using=(
            "CASE WHEN raw_content IS NULL THEN NULL "
            "WHEN jsonb_typeof(raw_content) = 'string' THEN raw_content #>> '{}' "
            "ELSE raw_content::text END"
        ),
    )

    op.create_table(
        "processing_jobs",
        sa.Column("job_id", sa.String(length=64), nullable=False),
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("job_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("current_stage", sa.String(length=32), nullable=False),
        sa.Column("input_path", sa.Text(), nullable=False),
        sa.Column("total_count", sa.Integer(), nullable=False),
        sa.Column("completed_count", sa.Integer(), nullable=False),
        sa.Column("failed_count", sa.Integer(), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("error_info", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.workspace_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("job_id"),
    )
    op.create_index(op.f("ix_processing_jobs_workspace_id"), "processing_jobs", ["workspace_id"])
    op.create_index(op.f("ix_processing_jobs_status"), "processing_jobs", ["status"])
    op.create_index(op.f("ix_processing_jobs_current_stage"), "processing_jobs", ["current_stage"])


def downgrade() -> None:
    op.drop_index(op.f("ix_processing_jobs_current_stage"), table_name="processing_jobs")
    op.drop_index(op.f("ix_processing_jobs_status"), table_name="processing_jobs")
    op.drop_index(op.f("ix_processing_jobs_workspace_id"), table_name="processing_jobs")
    op.drop_table("processing_jobs")

    op.alter_column(
        "assets",
        "raw_content",
        existing_type=sa.Text(),
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=True,
        postgresql_using="raw_content::jsonb",
    )
    op.drop_constraint("uq_asset_source_key", "assets", type_="unique")
    op.drop_column("assets", "asset_name_source")
    op.drop_column("assets", "content_hash")
    op.drop_column("assets", "asset_key")
    op.drop_column("assets", "file_type")

    op.drop_constraint("uq_source_file_workspace_relative_path", "source_files", type_="unique")
    op.create_unique_constraint(
        "uq_source_file_workspace_sha256",
        "source_files",
        ["workspace_id", "sha256"],
    )
