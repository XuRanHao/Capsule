"""Unit contracts for lease-fenced image/text CPU Asset processors."""

from __future__ import annotations

import asyncio
import hashlib
from urllib.parse import unquote, urlparse
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from capsule.enums import AssetType
from capsule.pipeline.asset_factory import AssetFactory
from capsule.pipeline.processing_task_processor import (
    DurableProcessingSource,
    ImageProcessingTaskProcessor,
    TextProcessingTaskProcessor,
)
from capsule.pipeline.video_task_runtime import (
    ProcessingTaskKind,
    ProcessingTaskLease,
    ProcessingTaskMessage,
    ProcessingTaskProgress,
    ResourceClass,
)
from capsule.schemas import AssetCreate, AssetDraft, DiscoveredFile, SourceContext


@dataclass
class _FencedCommitter:
    calls: list[tuple[ProcessingTaskMessage, ProcessingTaskLease, list[AssetCreate]]] = field(
        default_factory=list
    )

    async def commit_assets(
        self,
        message: ProcessingTaskMessage,
        lease: ProcessingTaskLease,
        assets: list[AssetCreate],
        *,
        source_sha256: str,
    ) -> list[str]:
        assert source_sha256
        self.calls.append((message, lease, assets))
        return [asset.asset_id for asset in assets]


class _ImageParser:
    def __init__(self) -> None:
        self.sources: list[DiscoveredFile] = []

    async def assetize(self, source_file: DiscoveredFile) -> list[AssetDraft]:
        self.sources.append(source_file)
        return [
            AssetDraft(
                asset_type=AssetType.IMAGE,
                file_name="image.png",
                source_locator={"type": "whole_file"},
                file_info={"width": 2, "height": 1},
            )
        ]


class _DocumentParser:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, object, dict[str, int]]] = []

    async def assetize_file(
        self,
        path: Path,
        token_counter: object,
        **options: int,
    ) -> list[AssetDraft]:
        self.calls.append((path, token_counter, options))
        return [
            AssetDraft(
                asset_type=AssetType.TEXT_BLOCK,
                file_name=path.name,
                source_locator={"type": "text_range", "block_index": 0},
                raw_content="hello CPU processor",
                file_info={"token_count": 3},
            )
        ]


async def _source_loader(message: ProcessingTaskMessage) -> DurableProcessingSource:
    return await asyncio.to_thread(_source_from_message, message)


def _source_from_message(message: ProcessingTaskMessage) -> DurableProcessingSource:
    parsed = urlparse(message.source_uri)
    raw_path = unquote(parsed.path if parsed.scheme else message.source_uri)
    if len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
        raw_path = raw_path[1:]
    path = Path(raw_path)
    return DurableProcessingSource(
        source_file=DiscoveredFile(
            path=str(path),
            relative_path=f"canonical/nested/{path.name}",
            extension=path.suffix,
            size_bytes=path.stat().st_size,
        ),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _message(kind: ProcessingTaskKind, path: Path) -> ProcessingTaskMessage:
    route_key = "cpu_image" if kind is ProcessingTaskKind.IMAGE else "cpu_text"
    return ProcessingTaskMessage(
        task_id=f"{kind.value}-task-1",
        job_id="job-1",
        workspace_id="workspace-1",
        source_file_id=f"source-{kind.value}",
        generation=2,
        result_version=3,
        source_uri=path.as_uri(),
        task_kind=kind,
        resource_class=ResourceClass.CPU,
        route_key=route_key,
    )


def _lease(message: ProcessingTaskMessage) -> ProcessingTaskLease:
    return ProcessingTaskLease(
        task_id=message.task_id or "",
        source_file_id=message.source_file_id,
        source_generation=message.generation,
        attempt=1,
        worker_id="cpu-worker",
        result_version=message.result_version,
        task_kind=message.task_kind,
        resource_class=ResourceClass.CPU,
        route_key=message.route_key,
        processor_version=message.processor_version,
        lease_token="db-fence",
    )


@pytest.mark.parametrize(
    ("kind", "suffix"),
    [(ProcessingTaskKind.IMAGE, ".png"), (ProcessingTaskKind.TEXT, ".md")],
)
async def test_cpu_processor_persists_one_fenced_asset_batch_and_reports_progress(
    tmp_path: Path,
    kind: ProcessingTaskKind,
    suffix: str,
) -> None:
    path = tmp_path / f"source{suffix}"
    path.write_text("payload", encoding="utf-8")
    committer = _FencedCommitter()
    image = _ImageParser()
    document = _DocumentParser()
    message = _message(kind, path)
    processor: ImageProcessingTaskProcessor | TextProcessingTaskProcessor
    if kind is ProcessingTaskKind.IMAGE:
        processor = ImageProcessingTaskProcessor(
            committer=committer,
            source_file_loader=_source_loader,
            image_parser=image,  # type: ignore[arg-type]
            asset_factory=AssetFactory(),
        )
    else:
        processor = TextProcessingTaskProcessor(
            committer=committer,
            source_file_loader=_source_loader,
            token_counter=object(),  # type: ignore[arg-type]
            document_parser=document,  # type: ignore[arg-type]
            asset_factory=AssetFactory(),
        )
    progress: list[ProcessingTaskProgress] = []

    async def collect(item: ProcessingTaskProgress) -> None:
        progress.append(item)

    result = await processor.process(message, _lease(message), collect)

    assert len(committer.calls) == 1
    committed_message, committed_lease, assets = committer.calls[0]
    assert committed_message == message
    assert committed_lease == _lease(message)
    assert [asset.generation for asset in assets] == [message.generation]
    assert [asset.workspace_id for asset in assets] == [message.workspace_id]
    assert result.metadata == {
        "source_file_id": message.source_file_id,
        "source_generation": 2,
        "asset_count": 1,
        "committed_asset_count": 1,
        "task_kind": kind.value,
    }
    assert [item.detail["stage"] for item in progress] == [
        "parsing",
        "assets_ready",
        "assets_persisted",
    ]
    if kind is ProcessingTaskKind.IMAGE:
        assert image.sources[0].relative_path == f"canonical/nested/{path.name}"
        assert document.calls == []
    else:
        assert image.sources == []
        assert document.calls[0][0] == path


async def test_cpu_processor_rejects_a_source_that_differs_from_its_durable_digest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"current source")
    message = _message(ProcessingTaskKind.IMAGE, path)
    committer = _FencedCommitter()

    async def stale_source_loader(
        observed_message: ProcessingTaskMessage,
    ) -> DurableProcessingSource:
        loaded = await _source_loader(observed_message)
        return DurableProcessingSource(source_file=loaded.source_file, sha256="0" * 64)

    processor = ImageProcessingTaskProcessor(
        committer=committer,
        source_file_loader=stale_source_loader,
        image_parser=_ImageParser(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="durable digest"):
        await processor.process(message, _lease(message), _discard_progress)

    assert committer.calls == []


async def test_cpu_processor_rechecks_digest_after_parsing_before_commit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"original source")
    message = _message(ProcessingTaskKind.IMAGE, path)
    committer = _FencedCommitter()

    class MutatingImageParser(_ImageParser):
        async def assetize(self, source_file: DiscoveredFile) -> list[AssetDraft]:
            path.write_bytes(b"replacement source")
            return await super().assetize(source_file)

    processor = ImageProcessingTaskProcessor(
        committer=committer,
        source_file_loader=_source_loader,
        image_parser=MutatingImageParser(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="changed while it was being parsed"):
        await processor.process(message, _lease(message), _discard_progress)

    assert committer.calls == []


async def test_cpu_processor_rejects_wrong_route_or_lease_before_any_asset_write(
    tmp_path: Path,
) -> None:
    path = tmp_path / "image.png"
    path.write_text("not read", encoding="utf-8")
    committer = _FencedCommitter()
    message = _message(ProcessingTaskKind.IMAGE, path)
    processor = ImageProcessingTaskProcessor(
        committer=committer,
        source_file_loader=_source_loader,
        image_parser=_ImageParser(),  # type: ignore[arg-type]
    )
    lease = _lease(message)
    wrong_route = ProcessingTaskLease(
        task_id=lease.task_id,
        source_file_id=lease.source_file_id,
        source_generation=lease.source_generation,
        attempt=lease.attempt,
        worker_id=lease.worker_id,
        result_version=lease.result_version,
        task_kind=lease.task_kind,
        resource_class=lease.resource_class,
        route_key="cpu_text",
        processor_version=lease.processor_version,
        lease_token=lease.lease_token,
    )

    with pytest.raises(ValueError, match="does not match"):
        await processor.process(message, wrong_route, _discard_progress)

    assert committer.calls == []


async def test_cpu_processor_adds_context_supplied_by_the_batch_coordinator(
    tmp_path: Path,
) -> None:
    path = tmp_path / "image.png"
    path.write_text("not read", encoding="utf-8")
    committer = _FencedCommitter()
    message = _message(ProcessingTaskKind.IMAGE, path)
    provided = [
        SourceContext(
            text="Mentioned by the neighbouring Markdown document.",
            relation_type="markdown_image_reference",
            source_path="notes/overview.md",
            heading_path=["Overview"],
        )
    ]

    async def load_contexts(
        observed_message: ProcessingTaskMessage,
        source_file: DiscoveredFile,
    ) -> list[SourceContext]:
        assert observed_message == message
        assert source_file.relative_path == f"canonical/nested/{path.name}"
        return provided

    processor = ImageProcessingTaskProcessor(
        committer=committer,
        source_file_loader=_source_loader,
        source_contexts_loader=load_contexts,
        image_parser=_ImageParser(),  # type: ignore[arg-type]
    )

    await processor.process(message, _lease(message), _discard_progress)

    assert committer.calls[0][2][0].source_contexts == provided


async def test_text_cpu_processor_accepts_the_same_batch_context_loader(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("text", encoding="utf-8")
    committer = _FencedCommitter()
    message = _message(ProcessingTaskKind.TEXT, path)
    provided = [SourceContext(text="batch context", relation_type="sibling_document")]

    async def load_contexts(
        _message: ProcessingTaskMessage,
        _source_file: DiscoveredFile,
    ) -> list[SourceContext]:
        return provided

    processor = TextProcessingTaskProcessor(
        committer=committer,
        source_file_loader=_source_loader,
        source_contexts_loader=load_contexts,
        token_counter=object(),  # type: ignore[arg-type]
        document_parser=_DocumentParser(),  # type: ignore[arg-type]
    )

    await processor.process(message, _lease(message), _discard_progress)

    assert committer.calls[0][2][0].source_contexts == provided


async def _discard_progress(_progress: ProcessingTaskProgress) -> None:
    return None
