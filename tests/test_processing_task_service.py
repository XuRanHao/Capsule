"""Construction seams for trusted CPU processing-task workers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from capsule.config import Settings
from capsule.db.repositories import PreparedJobProcessingTask, ProcessingTaskSourceInput
from capsule.parsers.document_media import RapidOcrEngine
from capsule.pipeline.processing_task_service import (
    BrowserProcessingTaskSubmissionService,
    _document_parser,
    postgres_source_contexts_loader,
    postgres_source_file_loader,
)
from capsule.pipeline.video_task_runtime import ProcessingTaskKind, ProcessingTaskMessage
from capsule.schemas import DiscoveredFile


class _Session:
    def __init__(self, source: object) -> None:
        self._source = source

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def scalar(self, _statement: object) -> object:
        return self._source


class _ObjectStorage:
    async def download_uri_to_file(self, uri: str, destination: Path) -> None:
        assert uri == "s3://capsule/sources/image.png"
        destination.write_bytes(b"remote source")


async def test_postgres_source_loader_uses_canonical_source_metadata_not_stream_uri() -> None:
    source = SimpleNamespace(
        workspace_id="workspace-1",
        processing_generation=4,
        storage_uri="s3://capsule/sources/image.png",
        relative_path="nested/image.png",
        file_size_bytes=123,
        sha256="a" * 64,
    )
    loader = postgres_source_file_loader(lambda: _Session(source), cast(object, _ObjectStorage()))
    message = ProcessingTaskMessage(
        task_id="image-task-1",
        job_id="job-1",
        workspace_id="workspace-1",
        source_file_id="source-1",
        generation=4,
        source_uri="file:///untrusted/stream/path.png",
    )

    loaded = await loader(message)

    assert Path(loaded.source_file.path).read_bytes() == b"remote source"
    assert loaded.source_file.relative_path == "nested/image.png"
    assert loaded.source_file.extension == ".png"
    assert loaded.source_file.size_bytes == 123
    assert loaded.sha256 == "a" * 64
    await loaded.cleanup()


async def test_postgres_source_loader_rejects_noncurrent_source_generation() -> None:
    source = SimpleNamespace(
        workspace_id="workspace-1",
        processing_generation=5,
        storage_uri="s3://capsule/sources/image.png",
        relative_path="nested/image.png",
        file_size_bytes=123,
        sha256="a" * 64,
    )
    loader = postgres_source_file_loader(lambda: _Session(source), cast(object, _ObjectStorage()))
    message = ProcessingTaskMessage(
        task_id="image-task-1",
        job_id="job-1",
        workspace_id="workspace-1",
        source_file_id="source-1",
        generation=4,
    )

    with pytest.raises(ValueError, match="canonical generation"):
        await loader(message)


@pytest.mark.asyncio
async def test_postgres_context_loader_reads_trusted_task_payload() -> None:
    loader = postgres_source_contexts_loader(
        lambda: _Session(
            {
                "source_contexts": [
                    {
                        "relation_type": "nearby_text",
                        "text": "A referenced frame.",
                        "source_path": "notes/note.md",
                    }
                ]
            }
        )
    )
    message = ProcessingTaskMessage(
        task_id="image-task-1",
        job_id="job-1",
        workspace_id="workspace-1",
        source_file_id="source-1",
        generation=1,
    )
    source = DiscoveredFile(
        path="/imports/image.png",
        relative_path="image.png",
        extension=".png",
        size_bytes=1,
    )

    contexts = await loader(message, source)

    assert len(contexts) == 1
    assert contexts[0].text == "A referenced frame."
    assert contexts[0].source_path == "notes/note.md"


@pytest.mark.parametrize("ocr_enabled", [False, True])
def test_cpu_text_worker_document_parser_matches_runner_media_and_ocr_settings(
    tmp_path: Path,
    ocr_enabled: bool,
) -> None:
    settings = Settings(
        document_media_root=tmp_path / "document-media",
        document_ocr_enabled=ocr_enabled,
        document_ocr_min_edge=41,
        document_ocr_min_area=4_321,
        document_ocr_min_confidence=0.71,
    )

    parser = _document_parser(settings)
    extractor = parser._media_extractor

    assert extractor is not None
    assert extractor._output_root == settings.document_media_root
    assert extractor._min_width == 41
    assert extractor._min_height == 41
    assert extractor._min_area == 4_321
    assert extractor._min_ocr_confidence == pytest.approx(0.71)
    assert isinstance(extractor._ocr_engine, RapidOcrEngine) is ocr_enabled


@pytest.mark.asyncio
async def test_browser_batch_uses_one_parent_job_and_three_trusted_routes(
    tmp_path: Path,
) -> None:
    image = tmp_path / "images" / "pic.png"
    note = tmp_path / "docs" / "note.md"
    video = tmp_path / "clip.mp4"
    image.parent.mkdir()
    note.parent.mkdir()
    image.write_bytes(b"not-decoded-by-submission")
    note.write_text("# Scene\n\nContext paragraph.\n\n![frame](../images/pic.png)")
    video.write_bytes(b"not-decoded-by-submission")
    sources = [
        DiscoveredFile(
            path=str(image),
            relative_path="images/pic.png",
            extension=".png",
            size_bytes=image.stat().st_size,
        ),
        DiscoveredFile(
            path=str(note),
            relative_path="docs/note.md",
            extension=".md",
            size_bytes=note.stat().st_size,
        ),
        DiscoveredFile(
            path=str(video),
            relative_path="clip.mp4",
            extension=".mp4",
            size_bytes=video.stat().st_size,
        ),
    ]

    class Repository:
        def __init__(self) -> None:
            self.items: list[ProcessingTaskSourceInput] = []
            self.dispatch_completed = False

        async def prepare_import_task_batch(
            self, **values: object
        ) -> list[PreparedJobProcessingTask]:
            assert values["job_id"] == "job-browser"
            assert values["post_asset_action"] == "enrich"
            supplied_items = values["items"]
            assert isinstance(supplied_items, list)
            self.items = cast(list[ProcessingTaskSourceInput], supplied_items)
            routes = [
                ("image", "cpu", "cpu_image"),
                ("text", "cpu", "cpu_text"),
                ("video", "mps_video", "mps_video"),
            ]
            return [
                PreparedJobProcessingTask(
                    job_id="job-browser",
                    source_file_id=f"source-{index}",
                    generation=1,
                    task_id=f"task-{index}",
                    task_kind=kind,
                    resource_class=resource,
                    route_key=route,
                    processor_version=1,
                    already_processed=False,
                    source_uri=self.items[index].source_uri,
                )
                for index, (kind, resource, route) in enumerate(routes)
            ]

        async def mark_import_dispatch_complete(self, *, job_id: str) -> None:
            assert job_id == "job-browser"
            self.dispatch_completed = True

    class Queue:
        def __init__(self, *, fail_publish: bool = False) -> None:
            self.messages: list[ProcessingTaskMessage] = []
            self.fail_publish = fail_publish

        async def start(self) -> None:
            return None

        async def publish(self, message: ProcessingTaskMessage) -> str:
            if self.fail_publish:
                raise ConnectionError("redis unavailable")
            self.messages.append(message)
            return "1-0"

    class TaskRepository:
        async def mark_published(self, _message: ProcessingTaskMessage) -> None:
            return None

    repository = Repository()
    queues = {kind: Queue() for kind in ProcessingTaskKind}
    service = BrowserProcessingTaskSubmissionService(
        settings=Settings(import_root=tmp_path),
        source_repository=repository,  # type: ignore[arg-type]
        queues=queues,  # type: ignore[arg-type]
        task_repositories={kind: TaskRepository() for kind in ProcessingTaskKind},
    )

    results = await service.submit_batch(
        job_id="job-browser",
        workspace_id="workspace-browser",
        source_files=sources,
        post_asset_action="enrich",
    )

    assert [result.route_key for result in results] == [
        "cpu_image",
        "cpu_text",
        "mps_video",
    ]
    assert [len(queues[kind].messages) for kind in ProcessingTaskKind] == [1, 1, 1, 0]
    image_input = repository.items[0]
    assert image_input.source_contexts
    assert queues[ProcessingTaskKind.IMAGE].messages[0].source_uri == image.resolve().as_uri()
    assert queues[ProcessingTaskKind.TEXT].messages[0].source_uri == note.resolve().as_uri()
    assert queues[ProcessingTaskKind.VIDEO].messages[0].source_uri == video.resolve().as_uri()


@pytest.mark.asyncio
async def test_browser_batch_leaves_durable_task_for_scheduler_when_publish_fails(
    tmp_path: Path,
) -> None:
    image = tmp_path / "pic.png"
    image.write_bytes(b"image")
    source = DiscoveredFile(
        path=str(image),
        relative_path="pic.png",
        extension=".png",
        size_bytes=image.stat().st_size,
    )

    class Repository:
        dispatch_completed = False

        async def prepare_import_task_batch(self, **_: object) -> list[PreparedJobProcessingTask]:
            return [
                PreparedJobProcessingTask(
                    job_id="job-browser",
                    source_file_id="source-image",
                    generation=1,
                    task_id="task-image",
                    task_kind="image",
                    resource_class="cpu",
                    route_key="cpu_image",
                    processor_version=1,
                    already_processed=False,
                )
            ]

        async def mark_import_dispatch_complete(self, *, job_id: str) -> None:
            assert job_id == "job-browser"
            self.dispatch_completed = True

    class FailedQueue:
        async def start(self) -> None:
            return None

        async def publish(self, _message: ProcessingTaskMessage) -> str:
            raise ConnectionError("redis unavailable")

    repository = Repository()
    service = BrowserProcessingTaskSubmissionService(
        settings=Settings(import_root=tmp_path),
        source_repository=repository,  # type: ignore[arg-type]
        queues={ProcessingTaskKind.IMAGE: FailedQueue()},  # type: ignore[dict-item]
        task_repositories={ProcessingTaskKind.IMAGE: object()},
    )

    results = await service.submit_batch(
        job_id="job-browser",
        workspace_id="workspace-browser",
        source_files=[source],
    )

    assert len(results) == 1
    assert not results[0].published
