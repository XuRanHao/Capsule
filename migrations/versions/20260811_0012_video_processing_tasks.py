"""Add durable, fenced whole-video processing task facts.

Revision ID: 20260811_0012
Revises: 20260810_0011
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260811_0012"
down_revision: str | None = "20260810_0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "video_processing_tasks",
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("parent_job_id", sa.String(length=64), nullable=False),
        sa.Column("source_file_id", sa.String(length=64), nullable=False),
        sa.Column("source_generation", sa.Integer(), nullable=False),
        sa.Column("result_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("owner_id", sa.String(length=255), nullable=True),
        sa.Column("message_id", sa.String(length=128), nullable=True),
        sa.Column(
            "progress",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("lease_deadline_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("progress_deadline_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("hard_deadline_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dlq_published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("parent_accounted_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["parent_job_id"], ["processing_jobs.job_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["source_file_id"], ["source_files.source_file_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("task_id"),
        sa.UniqueConstraint(
            "source_file_id",
            "source_generation",
            "result_version",
            name="uq_video_task_source_generation_result_version",
        ),
    )
    op.create_index(
        "ix_video_processing_tasks_parent_job_id",
        "video_processing_tasks",
        ["parent_job_id"],
    )
    op.create_index(
        "ix_video_processing_tasks_source_file_id",
        "video_processing_tasks",
        ["source_file_id"],
    )
    op.create_index("ix_video_processing_tasks_status", "video_processing_tasks", ["status"])
    op.create_index("ix_video_processing_tasks_stage", "video_processing_tasks", ["stage"])
    op.create_index("ix_video_processing_tasks_owner_id", "video_processing_tasks", ["owner_id"])
    op.create_index(
        "ix_video_processing_tasks_progress_deadline_at",
        "video_processing_tasks",
        ["progress_deadline_at"],
    )
    op.create_index(
        "ix_video_processing_tasks_hard_deadline_at",
        "video_processing_tasks",
        ["hard_deadline_at"],
    )
    op.create_index(
        "ix_video_processing_tasks_next_retry_at",
        "video_processing_tasks",
        ["next_retry_at"],
    )
    op.create_index(
        "ix_video_processing_tasks_last_published_at",
        "video_processing_tasks",
        ["last_published_at"],
    )
    op.create_index(
        "ix_video_processing_tasks_dlq_published_at",
        "video_processing_tasks",
        ["dlq_published_at"],
    )
    op.create_index(
        "ix_video_processing_tasks_status_next_retry",
        "video_processing_tasks",
        ["status", "next_retry_at"],
    )
    op.create_index(
        "ix_video_processing_tasks_parent_status",
        "video_processing_tasks",
        ["parent_job_id", "status"],
    )


def downgrade() -> None:
    op.drop_table("video_processing_tasks")
