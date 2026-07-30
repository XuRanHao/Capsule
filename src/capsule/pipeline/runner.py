"""Top-level file discovery, assetization, and PostgreSQL persistence."""

import asyncio
import hashlib
import json
import mimetypes
import posixpath
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, Field

from capsule.config import Settings, get_settings
from capsule.db.repositories import AssetRepository
from capsule.db.session import Database
from capsule.enums import PipelineStage
from capsule.model_clients.tokenization import ArkTokenCounter, TokenCounter
from capsule.parsers import discover_files
from capsule.parsers.assetizer import Assetizer
from capsule.parsers.discovery import sha256_file
from capsule.parsers.image import ImageParser
from capsule.parsers.markdown import MarkdownParser
from capsule.parsers.text import TextParser
from capsule.parsers.video import VideoParser, VideoSegmentationConfig, VisualEmbedder
from capsule.pipeline.asset_factory import AssetFactory
from capsule.pipeline.video_media import VideoArtifactStorage, VideoDerivedMediaWriter
from capsule.schemas import AssetDraft, DiscoveredFile, SourceContext
from capsule.storage.object_storage import ObjectStorage

AssetStoredCallback = Callable[[list[str]], Awaitable[None]]


class PipelinePlan(BaseModel):
    workspace_id: str
    input_path: str
    file_count: int
    total_bytes: int
    counts_by_extension: dict[str, int] = Field(default_factory=dict)


class PipelineRunResult(BaseModel):
    job_id: str
    workspace_id: str
    file_count: int
    succeeded_count: int
    failed_count: int
    asset_count: int
    asset_ids: list[str] = Field(default_factory=list)
    skipped_count: int = 0
    errors: list[dict[str, str]] = Field(default_factory=list)


@dataclass(slots=True, frozen=True)
class _FileOutcome:
    succeeded: bool
    asset_count: int = 0
    asset_ids: list[str] = field(default_factory=list)
    skipped: bool = False
    error: dict[str, str] | None = None
    stage_durations_ms: dict[str, float] = field(default_factory=dict)


class PipelineRunner:
    """Run deterministic parsers and atomically persist each source file's assets."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        database: Database | None = None,
        token_counter: TokenCounter | None = None,
        video_embedder: VisualEmbedder | None = None,
        object_storage: VideoArtifactStorage | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._database = database
        self._token_counter = token_counter
        self._video_embedder = video_embedder
        self._object_storage = object_storage

    def build_plan(self, input_path: Path, workspace_id: str) -> PipelinePlan:
        files = discover_files(input_path)
        counts = Counter(item.extension for item in files)
        return PipelinePlan(
            workspace_id=workspace_id,
            input_path=str(input_path.expanduser().resolve()),
            file_count=len(files),
            total_bytes=sum(item.size_bytes for item in files),
            counts_by_extension=dict(sorted(counts.items())),
        )

    async def run(
        self,
        input_path: Path,
        workspace_id: str,
        *,
        job_id: str | None = None,
        on_assets_stored: AssetStoredCallback | None = None,
        finalize_job: bool = True,
    ) -> PipelineRunResult:
        discovery_started = time.perf_counter()
        source_files = discover_files(input_path)
        stage_durations_ms = {
            PipelineStage.DISCOVERING.value: (
                time.perf_counter() - discovery_started
            )
            * 1000,
            PipelineStage.PARSING.value: 0.0,
            PipelineStage.SEGMENTING.value: 0.0,
            PipelineStage.ASSET_STORED.value: 0.0,
        }
        database = self._database or Database(self._settings)
        owns_database = self._database is None
        counter = self._token_counter
        owns_counter = False
        if counter is None and any(item.extension in {".md", ".txt"} for item in source_files):
            try:
                counter = ArkTokenCounter(self._settings)
                owns_counter = True
            except Exception:
                counter = None

        repository = AssetRepository(database)
        factory = AssetFactory()
        assetizer = _build_assetizer(counter, self._settings, self._video_embedder)
        image_source_contexts = await asyncio.to_thread(
            _collect_image_source_contexts,
            source_files,
        )
        object_storage = self._object_storage
        media_writer: VideoDerivedMediaWriter | None = None
        if any(item.extension in {".mp4", ".mov"} for item in source_files):
            object_storage = object_storage or ObjectStorage(self._settings)
            media_writer = VideoDerivedMediaWriter(
                object_storage,
                concurrency=self._settings.ffmpeg_concurrency,
            )
        if job_id is None:
            job_id = await repository.create_job(
                workspace_id=workspace_id,
                input_path=input_path,
                total_count=len(source_files),
            )
        try:
            processing_started = time.perf_counter()
            outcomes = await self._process_files(
                source_files=source_files,
                workspace_id=workspace_id,
                job_id=job_id,
                repository=repository,
                factory=factory,
                assetizer=assetizer,
                media_writer=media_writer,
                image_source_contexts=image_source_contexts,
                on_assets_stored=on_assets_stored,
            )
            processing_elapsed_ms = (
                time.perf_counter() - processing_started
            ) * 1000
            raw_durations = {
                stage: sum(
                    outcome.stage_durations_ms.get(stage, 0.0)
                    for outcome in outcomes
                )
                for stage in (
                    PipelineStage.PARSING.value,
                    PipelineStage.SEGMENTING.value,
                    PipelineStage.ASSET_STORED.value,
                )
            }
            raw_total_ms = sum(raw_durations.values())
            if raw_total_ms:
                for stage, duration_ms in raw_durations.items():
                    stage_durations_ms[stage] = (
                        processing_elapsed_ms * duration_ms / raw_total_ms
                    )
            await repository.add_job_stage_durations(
                job_id=job_id,
                durations_ms=stage_durations_ms,
            )
            if finalize_job:
                await repository.finalize_job(job_id=job_id)
            errors = [outcome.error for outcome in outcomes if outcome.error is not None]
            asset_ids = [
                asset_id
                for outcome in outcomes
                for asset_id in outcome.asset_ids
            ]
            return PipelineRunResult(
                job_id=job_id,
                workspace_id=workspace_id,
                file_count=len(source_files),
                succeeded_count=sum(outcome.succeeded for outcome in outcomes),
                failed_count=sum(not outcome.succeeded for outcome in outcomes),
                asset_count=sum(outcome.asset_count for outcome in outcomes),
                asset_ids=asset_ids,
                skipped_count=sum(outcome.skipped for outcome in outcomes),
                errors=errors,
            )
        finally:
            if owns_counter and isinstance(counter, ArkTokenCounter):
                await counter.close()
            if owns_database:
                await database.dispose()

    async def _process_files(
        self,
        *,
        source_files: list[DiscoveredFile],
        workspace_id: str,
        job_id: str,
        repository: AssetRepository,
        factory: AssetFactory,
        assetizer: Assetizer,
        media_writer: VideoDerivedMediaWriter | None,
        image_source_contexts: Mapping[str, list[SourceContext]],
        on_assets_stored: AssetStoredCallback | None,
    ) -> list[_FileOutcome]:
        """Process files with a fixed worker count and deterministic result ordering."""
        if not source_files:
            return []

        outcomes: list[_FileOutcome | None] = [None] * len(source_files)
        next_index = 0

        async def worker() -> None:
            nonlocal next_index
            while next_index < len(source_files):
                index = next_index
                next_index += 1
                outcomes[index] = await self._process_file(
                    source_file=source_files[index],
                    workspace_id=workspace_id,
                    job_id=job_id,
                    repository=repository,
                    factory=factory,
                    assetizer=assetizer,
                    media_writer=media_writer,
                    source_contexts=image_source_contexts.get(
                        source_files[index].relative_path,
                        [],
                    ),
                    on_assets_stored=on_assets_stored,
                )

        workers = [
            asyncio.create_task(worker())
            for _ in range(min(self._settings.file_parse_concurrency, len(source_files)))
        ]
        try:
            await asyncio.gather(*workers)
        except BaseException:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise
        if any(outcome is None for outcome in outcomes):
            raise RuntimeError("file processing ended before all files completed")
        return [outcome for outcome in outcomes if outcome is not None]

    async def _process_file(
        self,
        *,
        source_file: DiscoveredFile,
        workspace_id: str,
        job_id: str,
        repository: AssetRepository,
        factory: AssetFactory,
        assetizer: Assetizer,
        media_writer: VideoDerivedMediaWriter | None,
        source_contexts: list[SourceContext],
        on_assets_stored: AssetStoredCallback | None,
    ) -> _FileOutcome:
        source_file_id: str | None = None
        stage_durations_ms = {
            PipelineStage.PARSING.value: 0.0,
            PipelineStage.SEGMENTING.value: 0.0,
            PipelineStage.ASSET_STORED.value: 0.0,
        }
        try:
            phase_started = time.perf_counter()
            try:
                digest = await asyncio.to_thread(
                    sha256_file,
                    Path(source_file.path),
                )
                prepared = await repository.prepare_source_file(
                    workspace_id=workspace_id,
                    source_file=source_file,
                    sha256=digest,
                    mime_type=_mime_type(source_file),
                    processing_fingerprint=_processing_fingerprint(
                        source_file,
                        self._settings,
                        source_contexts=source_contexts,
                    ),
                )
            finally:
                stage_durations_ms[PipelineStage.PARSING.value] += (
                    time.perf_counter() - phase_started
                ) * 1000
            source_file_id = prepared.source_file_id
            if prepared.already_processed:
                phase_started = time.perf_counter()
                await repository.record_file_success(job_id=job_id)
                stage_durations_ms[PipelineStage.ASSET_STORED.value] += (
                    time.perf_counter() - phase_started
                ) * 1000
                return _FileOutcome(
                    succeeded=True,
                    asset_count=prepared.asset_count,
                    skipped=True,
                    stage_durations_ms=stage_durations_ms,
                )
            phase_started = time.perf_counter()
            try:
                result = await assetizer.assetize(source_file)
            finally:
                stage_durations_ms[PipelineStage.SEGMENTING.value] += (
                    time.perf_counter() - phase_started
                ) * 1000
            if not result.succeeded:
                raise ValueError(result.error_message or "assetization failed")
            if source_contexts:
                result.assets = [
                    draft.model_copy(
                        update={
                            "source_contexts": [
                                *draft.source_contexts,
                                *source_contexts,
                            ]
                        }
                    )
                    for draft in result.assets
                ]
            phase_started = time.perf_counter()
            try:
                assets = factory.build_many(
                    workspace_id=workspace_id,
                    source_file_id=source_file_id,
                    source_sha256=digest,
                    source_file=source_file,
                    drafts=result.assets,
                )
                if any(asset.asset_type.value == "video_segment" for asset in assets):
                    if media_writer is None:
                        raise RuntimeError("video media writer is unavailable")
                    assets = await media_writer.persist(
                        source_file=source_file,
                        assets=assets,
                    )
                stored = await repository.replace_assets(
                    source_file_id=source_file_id,
                    assets=assets,
                )
                if on_assets_stored is not None and stored.asset_ids:
                    await on_assets_stored(stored.asset_ids)
                await repository.record_file_success(job_id=job_id)
            finally:
                stage_durations_ms[PipelineStage.ASSET_STORED.value] += (
                    time.perf_counter() - phase_started
                ) * 1000
            return _FileOutcome(
                succeeded=True,
                asset_count=len(stored.asset_ids),
                asset_ids=stored.asset_ids,
                stage_durations_ms=stage_durations_ms,
            )
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            await repository.record_file_failure(
                job_id=job_id,
                source_file_id=source_file_id,
                relative_path=source_file.relative_path,
                error=message,
            )
            return _FileOutcome(
                succeeded=False,
                error={"relative_path": source_file.relative_path, "error": message},
                stage_durations_ms=stage_durations_ms,
            )


def _build_assetizer(
    counter: TokenCounter | None,
    settings: Settings,
    video_embedder: VisualEmbedder | None,
) -> Assetizer:
    markdown = MarkdownParser()
    plain_text = TextParser()
    image = ImageParser()
    video = VideoParser(
        concurrency=settings.ffmpeg_concurrency,
        config=VideoSegmentationConfig(
            scene_threshold=settings.video_scene_threshold,
            min_scene_seconds=settings.video_min_scene_seconds,
            max_shot_seconds=settings.video_max_shot_seconds,
            window_seconds=settings.video_window_seconds,
            sample_interval_seconds=settings.video_sample_interval_seconds,
            max_candidate_frames=settings.video_max_candidate_frames,
            max_representative_frames=settings.video_max_representative_frames,
        ),
        embedder=video_embedder,
        mobileclip_model_path=Path(settings.mobileclip_model_path),
        mobileclip_batch_size=settings.mobileclip_batch_size,
    )

    async def markdown_handler(source_file: DiscoveredFile) -> list[AssetDraft]:
        if counter is None:
            raise ValueError("CAPSULE_ARK_API_KEY is required to split text documents")
        return await markdown.assetize_file(
            Path(source_file.path),
            counter,
            max_tokens=settings.document_chunk_max_tokens,
        )

    async def plain_text_handler(source_file: DiscoveredFile) -> list[AssetDraft]:
        if counter is None:
            raise ValueError("CAPSULE_ARK_API_KEY is required to split text documents")
        return await plain_text.assetize_file(
            Path(source_file.path),
            counter,
            max_tokens=settings.document_chunk_max_tokens,
        )

    return Assetizer(
        {
            ".md": markdown_handler,
            ".txt": plain_text_handler,
            ".jpg": image.assetize,
            ".jpeg": image.assetize,
            ".png": image.assetize,
            ".webp": image.assetize,
            ".mp4": video.assetize,
            ".mov": video.assetize,
        }
    )


def _mime_type(source_file: DiscoveredFile) -> str:
    guessed, _ = mimetypes.guess_type(source_file.path)
    return guessed or "application/octet-stream"


def _processing_fingerprint(
    source_file: DiscoveredFile,
    settings: Settings,
    *,
    source_contexts: list[SourceContext] | None = None,
) -> str:
    payload: dict[str, object] = {
        "version": settings.assetization_version,
        "extension": source_file.extension,
    }
    if source_file.extension in {".md", ".txt"}:
        payload["document_chunk_max_tokens"] = settings.document_chunk_max_tokens
    elif source_file.extension in {".mp4", ".mov"}:
        payload["video"] = {
            "scene_threshold": settings.video_scene_threshold,
            "min_scene_seconds": settings.video_min_scene_seconds,
            "max_shot_seconds": settings.video_max_shot_seconds,
            "window_seconds": settings.video_window_seconds,
            "sample_interval_seconds": settings.video_sample_interval_seconds,
            "max_candidate_frames": settings.video_max_candidate_frames,
            "max_representative_frames": settings.video_max_representative_frames,
            "mobileclip_model_path": settings.mobileclip_model_path,
        }
    if source_contexts:
        payload["source_contexts"] = [
            context.model_dump(mode="json") for context in source_contexts
        ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _collect_image_source_contexts(
    source_files: list[DiscoveredFile],
) -> dict[str, list[SourceContext]]:
    """Map local Markdown image references to their associated text paragraphs."""
    available_paths = {source.relative_path for source in source_files}
    contexts_by_image: dict[str, list[SourceContext]] = {}
    parser = MarkdownParser()
    for source_file in source_files:
        if source_file.extension != ".md":
            continue
        try:
            parsed = parser.parse_file(Path(source_file.path))
        except (OSError, UnicodeError):
            continue
        document_title = (
            next(
                (
                    reference.heading_path[0]
                    for reference in parsed.image_references
                    if reference.heading_path
                ),
                None,
            )
            or Path(source_file.relative_path).stem
        )
        for reference in parsed.image_references:
            image_path = _resolve_markdown_image_path(
                document_path=source_file.relative_path,
                image_reference=reference.image_path,
            )
            if image_path is None or image_path not in available_paths:
                continue
            enriched = [
                context.model_copy(
                    update={
                        "paragraph_id": (
                            f"{source_file.relative_path}#block-{context.text_block_index}"
                            if context.text_block_index is not None
                            else None
                        ),
                        "source_path": source_file.relative_path,
                        "document_title": document_title,
                        "heading_path": list(reference.heading_path),
                    }
                )
                for context in reference.contexts
            ]
            existing = contexts_by_image.setdefault(image_path, [])
            for context in enriched:
                if context not in existing:
                    existing.append(context)
    return contexts_by_image


def _resolve_markdown_image_path(
    *,
    document_path: str,
    image_reference: str,
) -> str | None:
    parsed = urlparse(image_reference)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    decoded = unquote(parsed.path).replace("\\", "/")
    if decoded.startswith("/"):
        return None
    resolved = posixpath.normpath(
        posixpath.join(posixpath.dirname(document_path), decoded)
    )
    if resolved == ".." or resolved.startswith("../"):
        return None
    return resolved
