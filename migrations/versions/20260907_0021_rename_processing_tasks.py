"""Rename the durable task table to its generic name.

Revision ID: 20260907_0021
Revises: 20260814_0020
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260907_0021"
down_revision: str | None = "20260814_0020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_INDEX_RENAMES = (
    ("ix_video_processing_tasks_parent_job_id", "ix_processing_tasks_parent_job_id"),
    ("ix_video_processing_tasks_source_file_id", "ix_processing_tasks_source_file_id"),
    ("ix_video_processing_tasks_status", "ix_processing_tasks_status"),
    ("ix_video_processing_tasks_stage", "ix_processing_tasks_stage"),
    ("ix_video_processing_tasks_owner_id", "ix_processing_tasks_owner_id"),
    ("ix_video_processing_tasks_progress_deadline_at", "ix_processing_tasks_progress_deadline_at"),
    ("ix_video_processing_tasks_hard_deadline_at", "ix_processing_tasks_hard_deadline_at"),
    ("ix_video_processing_tasks_next_retry_at", "ix_processing_tasks_next_retry_at"),
    ("ix_video_processing_tasks_last_published_at", "ix_processing_tasks_last_published_at"),
    ("ix_video_processing_tasks_dlq_published_at", "ix_processing_tasks_dlq_published_at"),
    ("ix_video_processing_tasks_status_next_retry", "ix_processing_tasks_status_next_retry"),
    ("ix_video_processing_tasks_parent_status", "ix_processing_tasks_parent_status"),
    ("ix_video_processing_tasks_task_kind", "ix_processing_tasks_task_kind"),
    ("ix_video_processing_tasks_resource_class", "ix_processing_tasks_resource_class"),
    ("ix_video_processing_tasks_kind_status_next_retry", "ix_processing_tasks_kind_status_next_retry"),
    ("ix_video_processing_tasks_kind_identity", "ix_processing_tasks_kind_identity"),
)


def upgrade() -> None:
    op.rename_table("video_processing_tasks", "processing_tasks")
    op.execute(
        "ALTER TABLE processing_tasks RENAME CONSTRAINT "
        "uq_video_task_source_generation_result_version "
        "TO uq_processing_task_source_generation_result_version"
    )
    for old_name, new_name in _INDEX_RENAMES:
        op.execute(f"ALTER INDEX {old_name} RENAME TO {new_name}")


def downgrade() -> None:
    for old_name, new_name in reversed(_INDEX_RENAMES):
        op.execute(f"ALTER INDEX {new_name} RENAME TO {old_name}")
    op.execute(
        "ALTER TABLE processing_tasks RENAME CONSTRAINT "
        "uq_processing_task_source_generation_result_version "
        "TO uq_video_task_source_generation_result_version"
    )
    op.rename_table("processing_tasks", "video_processing_tasks")
