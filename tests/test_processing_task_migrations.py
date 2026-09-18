from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

from capsule.db.video_tasks import VideoProcessingTask


def test_processing_task_dispatch_round_is_backed_by_the_current_migration_head() -> None:
    config = Config(str(Path("alembic.ini").resolve()))
    script = ScriptDirectory.from_config(config)

    assert script.get_current_head() == "20260918_0035"
    assert "dispatch_round" in VideoProcessingTask.__table__.columns
