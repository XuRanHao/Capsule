"""Lease-fenced CPU processors for image and document Asset generations.

Parsing is side-effect free. The injected batch committer owns the one durable
transaction that validates the lease/source generation, writes the whole Asset
generation, marks the source complete and records task/parent completion.
Understanding, embedding, indexing and clustering are intentionally not called.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from capsule.model_clients.tokenization import TokenCounter
from capsule.parsers.discovery import sha256_file
from capsule.parsers.document import DocumentParser
from capsule.parsers.image import ImageParser
from capsule.pipeline.asset_factory import AssetFactory
from capsule.pipeline.video_task_runtime import (
    ProcessingTaskKind,
    ProcessingTaskLease,
    ProcessingTaskMessage,
    ProcessingTaskProgress,
    ProcessingTaskResult,
    ResourceClass,
)
from capsule.schemas import AssetCreate, AssetDraft, DiscoveredFile, SourceContext


@dataclass(frozen=True, slots=True)
class DurableProcessingSource:
    """Canonical source metadata plus the digest recorded at submission time."""

    source_file: DiscoveredFile
    sha256: str
    cleanup: Callable[[], Awaitable[None]] | None = None


SourceFileLoader = Callable[[ProcessingTaskMessage], Awaitable[DurableProcessingSource]]
SourceContextsLoader = Callable[
    [ProcessingTaskMessage, DiscoveredFile], Awaitable[list[SourceContext]]
]


class FencedAssetBatchCommitter(Protocol):
    """The only write boundary for one CPU task's complete Asset generation."""

    async def commit_assets(
        self,
        message: ProcessingTaskMessage,
        lease: ProcessingTaskLease,
        assets: list[AssetCreate],
        *,
        source_sha256: str,
    ) -> list[str]: ...


class _CPUAssetProcessor:
    def __init__(
        self,
        *,
        task_kind: ProcessingTaskKind,
        route_key: str,
        committer: FencedAssetBatchCommitter,
        source_file_loader: SourceFileLoader,
        source_contexts_loader: SourceContextsLoader | None = None,
        asset_factory: AssetFactory | None = None,
    ) -> None:
        self._task_kind = task_kind
        self._route_key = route_key
        self._committer = committer
        self._source_file_loader = source_file_loader
        self._source_contexts_loader = source_contexts_loader
        self._asset_factory = asset_factory or AssetFactory()

    async def _process_drafts(
        self,
        message: ProcessingTaskMessage,
        lease: ProcessingTaskLease,
        report_progress: Callable[[ProcessingTaskProgress], Awaitable[None]],
        parse: Callable[[DiscoveredFile], Awaitable[list[AssetDraft]]],
    ) -> ProcessingTaskResult:
        _assert_cpu_message_lease(
            message,
            lease,
            expected_kind=self._task_kind,
            expected_route_key=self._route_key,
        )
        await report_progress(ProcessingTaskProgress(detail={"stage": "parsing"}))
        durable_source = await self._source_file_loader(message)
        try:
            return await self._process_materialized_source(
                message=message,
                lease=lease,
                report_progress=report_progress,
                parse=parse,
                durable_source=durable_source,
            )
        finally:
            if durable_source.cleanup is not None:
                await durable_source.cleanup()

    async def _process_materialized_source(
        self,
        *,
        message: ProcessingTaskMessage,
        lease: ProcessingTaskLease,
        report_progress: Callable[[ProcessingTaskProgress], Awaitable[None]],
        parse: Callable[[DiscoveredFile], Awaitable[list[AssetDraft]]],
        durable_source: DurableProcessingSource,
    ) -> ProcessingTaskResult:
        source_file = durable_source.source_file
        expected_source_sha256 = durable_source.sha256
        source_sha256 = await asyncio.to_thread(sha256_file, Path(source_file.path))
        if source_sha256 != expected_source_sha256:
            raise ValueError("CPU processing source content no longer matches its durable digest")
        drafts = await parse(source_file)
        if not drafts:
            raise ValueError("CPU processing task produced no Assets")
        if self._source_contexts_loader is not None:
            source_contexts = await self._source_contexts_loader(message, source_file)
            if source_contexts:
                drafts = [
                    draft.model_copy(
                        update={
                            "source_contexts": [*draft.source_contexts, *source_contexts]
                        }
                    )
                    for draft in drafts
                ]
        source_sha256 = await asyncio.to_thread(sha256_file, Path(source_file.path))
        if source_sha256 != expected_source_sha256:
            raise ValueError("CPU processing source changed while it was being parsed")
        assets = self._asset_factory.build_many(
            workspace_id=message.workspace_id,
            source_file_id=message.source_file_id,
            source_sha256=source_sha256,
            source_file=source_file,
            drafts=drafts,
            generation=message.generation,
        )
        await report_progress(
            ProcessingTaskProgress(total_units=len(assets), detail={"stage": "assets_ready"})
        )
        committed_asset_ids = await self._committer.commit_assets(
            message,
            lease,
            assets,
            source_sha256=source_sha256,
        )
        if len(committed_asset_ids) != len(assets):
            raise RuntimeError("fenced CPU batch commit returned an unexpected Asset count")
        await report_progress(
            ProcessingTaskProgress(
                completed_units=len(assets),
                total_units=len(assets),
                detail={"stage": "assets_persisted"},
            )
        )
        return ProcessingTaskResult(
            result_ref=(
                f"processing-task://{message.task_kind.value}/"
                f"{message.task_id}/generation/{message.generation}"
            ),
            metadata={
                "source_file_id": message.source_file_id,
                "source_generation": message.generation,
                "asset_count": len(assets),
                "committed_asset_count": len(committed_asset_ids),
                "task_kind": message.task_kind.value,
            },
        )


class ImageProcessingTaskProcessor(_CPUAssetProcessor):
    """Asset-only processor trusted exclusively for the ``cpu_image`` route."""

    def __init__(
        self,
        *,
        committer: FencedAssetBatchCommitter,
        source_file_loader: SourceFileLoader,
        source_contexts_loader: SourceContextsLoader | None = None,
        image_parser: ImageParser | None = None,
        asset_factory: AssetFactory | None = None,
    ) -> None:
        super().__init__(
            task_kind=ProcessingTaskKind.IMAGE,
            route_key="cpu_image",
            committer=committer,
            source_file_loader=source_file_loader,
            source_contexts_loader=source_contexts_loader,
            asset_factory=asset_factory,
        )
        self._image_parser = image_parser or ImageParser()

    async def process(
        self,
        message: ProcessingTaskMessage,
        lease: ProcessingTaskLease,
        report_progress: Callable[[ProcessingTaskProgress], Awaitable[None]],
    ) -> ProcessingTaskResult:
        return await self._process_drafts(
            message,
            lease,
            report_progress,
            self._image_parser.assetize,
        )


class TextProcessingTaskProcessor(_CPUAssetProcessor):
    """Asset-only processor trusted exclusively for the ``cpu_text`` route."""

    def __init__(
        self,
        *,
        committer: FencedAssetBatchCommitter,
        source_file_loader: SourceFileLoader,
        source_contexts_loader: SourceContextsLoader | None = None,
        token_counter: TokenCounter,
        document_parser: DocumentParser | None = None,
        asset_factory: AssetFactory | None = None,
        min_tokens: int = 250,
        target_tokens: int = 400,
        max_tokens: int = 500,
        merge_max_tokens: int = 600,
        parent_max_tokens: int = 2_000,
    ) -> None:
        super().__init__(
            task_kind=ProcessingTaskKind.TEXT,
            route_key="cpu_text",
            committer=committer,
            source_file_loader=source_file_loader,
            source_contexts_loader=source_contexts_loader,
            asset_factory=asset_factory,
        )
        self._token_counter = token_counter
        self._document_parser = document_parser or DocumentParser()
        self._document_options = {
            "min_tokens": min_tokens,
            "target_tokens": target_tokens,
            "max_tokens": max_tokens,
            "merge_max_tokens": merge_max_tokens,
            "parent_max_tokens": parent_max_tokens,
        }

    async def process(
        self,
        message: ProcessingTaskMessage,
        lease: ProcessingTaskLease,
        report_progress: Callable[[ProcessingTaskProgress], Awaitable[None]],
    ) -> ProcessingTaskResult:
        async def parse(source_file: DiscoveredFile) -> list[AssetDraft]:
            return await self._document_parser.assetize_file(
                Path(source_file.path),
                self._token_counter,
                **self._document_options,
            )

        return await self._process_drafts(message, lease, report_progress, parse)


def _assert_cpu_message_lease(
    message: ProcessingTaskMessage,
    lease: ProcessingTaskLease,
    *,
    expected_kind: ProcessingTaskKind,
    expected_route_key: str,
) -> None:
    if (
        message.task_kind is not expected_kind
        or message.resource_class is not ResourceClass.CPU
        or message.route_key != expected_route_key
        or lease.task_kind is not expected_kind
        or lease.resource_class is not ResourceClass.CPU
        or lease.route_key != expected_route_key
        or message.task_id != lease.task_id
        or message.source_file_id != lease.source_file_id
        or message.generation != lease.source_generation
        or message.result_version != lease.result_version
        or message.processor_version != lease.processor_version
    ):
        raise ValueError("CPU processing task message does not match its database lease")
