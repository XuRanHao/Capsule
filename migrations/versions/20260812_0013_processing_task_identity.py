"""Add generic processor identity metadata to durable tasks.

Revision ID: 20260812_0013
Revises: 20260811_0012
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260812_0013"
down_revision: str | None = "20260811_0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "processing_jobs",
        sa.Column("post_asset_action", sa.String(length=32), server_default="none", nullable=False),
    )
    op.add_column(
        "processing_jobs",
        sa.Column("dispatch_completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "processing_jobs",
        sa.Column("assetization_completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "processing_jobs",
        sa.Column("workflow_owner_id", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "processing_jobs",
        sa.Column("workflow_lease_token", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "processing_jobs",
        sa.Column("workflow_lease_deadline_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_processing_jobs_assetization_completed_at",
        "processing_jobs",
        ["assetization_completed_at"],
    )
    op.create_index(
        "ix_processing_jobs_workflow_owner_id",
        "processing_jobs",
        ["workflow_owner_id"],
    )
    op.create_index(
        "ix_processing_jobs_workflow_lease_deadline_at",
        "processing_jobs",
        ["workflow_lease_deadline_at"],
    )
    op.add_column(
        "video_processing_tasks",
        sa.Column("task_kind", sa.String(length=32), server_default="video", nullable=False),
    )
    op.add_column(
        "video_processing_tasks",
        sa.Column(
            "resource_class",
            sa.String(length=32),
            server_default="mps_video",
            nullable=False,
        ),
    )
    op.add_column(
        "video_processing_tasks",
        sa.Column("processor_version", sa.Integer(), server_default="1", nullable=False),
    )
    op.add_column(
        "video_processing_tasks",
        sa.Column("route_key", sa.String(length=128), server_default="mps_video", nullable=False),
    )
    op.add_column(
        "video_processing_tasks",
        sa.Column("lease_token", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "video_processing_tasks",
        sa.Column(
            "input_payload",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_video_processing_tasks_task_kind",
        "video_processing_tasks",
        ["task_kind"],
    )
    op.create_index(
        "ix_video_processing_tasks_resource_class",
        "video_processing_tasks",
        ["resource_class"],
    )
    op.create_index(
        "ix_video_processing_tasks_kind_status_next_retry",
        "video_processing_tasks",
        ["task_kind", "status", "next_retry_at"],
    )
    # Keep the deployed video uniqueness constraint for rolling compatibility.
    # A later migration may replace it with a partial unique index once non-video
    # task kinds are actually written.
    op.create_index(
        "ix_video_processing_tasks_kind_identity",
        "video_processing_tasks",
        ["source_file_id", "source_generation", "task_kind", "result_version"],
    )


def downgrade() -> None:
    op.drop_index("ix_video_processing_tasks_kind_identity")
    op.drop_index("ix_video_processing_tasks_kind_status_next_retry")
    op.drop_index("ix_video_processing_tasks_resource_class")
    op.drop_index("ix_video_processing_tasks_task_kind")
    op.drop_column("video_processing_tasks", "input_payload")
    op.drop_column("video_processing_tasks", "lease_token")
    op.drop_column("video_processing_tasks", "route_key")
    op.drop_column("video_processing_tasks", "processor_version")
    op.drop_column("video_processing_tasks", "resource_class")
    op.drop_column("video_processing_tasks", "task_kind")
    op.drop_index("ix_processing_jobs_workflow_lease_deadline_at")
    op.drop_index("ix_processing_jobs_workflow_owner_id")
    op.drop_index("ix_processing_jobs_assetization_completed_at")
    op.drop_column("processing_jobs", "workflow_lease_deadline_at")
    op.drop_column("processing_jobs", "workflow_lease_token")
    op.drop_column("processing_jobs", "workflow_owner_id")
    op.drop_column("processing_jobs", "assetization_completed_at")
    op.drop_column("processing_jobs", "dispatch_completed_at")
    op.drop_column("processing_jobs", "post_asset_action")
