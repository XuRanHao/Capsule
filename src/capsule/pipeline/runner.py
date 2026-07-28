"""Top-level file discovery, assetization, and PostgreSQL persistence."""

import asyncio
import mimetypes
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from capsule.config import Settings, get_settings
from capsule.db.repositories import AssetRepository
from capsule.db.session import Database
from capsule.model_clients.tokenization import ArkTokenCounter, TokenCounter
from capsule.parsers import discover_files
from capsule.parsers.assetizer import Assetizer
from capsule.parsers.discovery import sha256_file
from capsule.parsers.image import ImageParser
from capsule.parsers.markdown import MarkdownParser
from capsule.pipeline.asset_factory import AssetFactory
from capsule.schemas import AssetDraft, DiscoveredFile


class PipelineNotReadyError(RuntimeError):
    pass


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
    errors: list[dict[str, str]] = Field(default_factory=list)


class PipelineRunner:
    """Run deterministic parsers and atomically persist each source file's assets."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        database: Database | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._database = database
        self._token_counter = token_counter

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

    async def run(self, input_path: Path, workspace_id: str) -> PipelineRunResult:
        source_files = discover_files(input_path)
        database = self._database or Database(self._settings)
        owns_database = self._database is None
        counter = self._token_counter
        owns_counter = False
        if counter is None and any(item.extension == ".md" for item in source_files):
            try:
                counter = ArkTokenCounter(self._settings)
                owns_counter = True
            except Exception:
                counter = None

        repository = AssetRepository(database)
        factory = AssetFactory()
        assetizer = _build_assetizer(counter)
        job_id = await repository.create_job(
            workspace_id=workspace_id,
            input_path=input_path,
            total_count=len(source_files),
        )
        succeeded = 0
        failed = 0
        asset_count = 0
        errors: list[dict[str, str]] = []
        try:
            for source_file in source_files:
                source_file_id: str | None = None
                try:
                    digest = await asyncio.to_thread(sha256_file, Path(source_file.path))
                    source_file_id = await repository.get_or_create_source_file(
                        workspace_id=workspace_id,
                        source_file=source_file,
                        sha256=digest,
                        mime_type=_mime_type(source_file),
                    )
                    result = await assetizer.assetize(source_file)
                    if not result.succeeded:
                        raise ValueError(result.error_message or "assetization failed")
                    assets = factory.build_many(
                        workspace_id=workspace_id,
                        source_file_id=source_file_id,
                        source_sha256=digest,
                        source_file=source_file,
                        drafts=result.assets,
                    )
                    stored = await repository.replace_assets(
                        source_file_id=source_file_id,
                        assets=assets,
                    )
                    await repository.record_file_success(job_id=job_id)
                    succeeded += 1
                    asset_count += len(stored.asset_ids)
                except Exception as exc:
                    message = str(exc) or type(exc).__name__
                    await repository.record_file_failure(
                        job_id=job_id,
                        source_file_id=source_file_id,
                        relative_path=source_file.relative_path,
                        error=message,
                    )
                    errors.append({"relative_path": source_file.relative_path, "error": message})
                    failed += 1
            await repository.finalize_job(job_id=job_id)
            return PipelineRunResult(
                job_id=job_id,
                workspace_id=workspace_id,
                file_count=len(source_files),
                succeeded_count=succeeded,
                failed_count=failed,
                asset_count=asset_count,
                errors=errors,
            )
        finally:
            if owns_counter and isinstance(counter, ArkTokenCounter):
                await counter.close()
            if owns_database:
                await database.dispose()


def _build_assetizer(counter: TokenCounter | None) -> Assetizer:
    markdown = MarkdownParser()
    image = ImageParser()

    async def markdown_handler(source_file: DiscoveredFile) -> list[AssetDraft]:
        if counter is None:
            raise ValueError("CAPSULE_ARK_API_KEY is required to split Markdown")
        return await markdown.assetize_file(Path(source_file.path), counter)

    async def video_handler(source_file: DiscoveredFile) -> list[AssetDraft]:
        raise PipelineNotReadyError(
            f"video visual segmentation is not implemented yet: {source_file.relative_path}"
        )

    return Assetizer(
        {
            ".md": markdown_handler,
            ".jpg": image.assetize,
            ".jpeg": image.assetize,
            ".png": image.assetize,
            ".webp": image.assetize,
            ".mp4": video_handler,
            ".mov": video_handler,
        }
    )


def _mime_type(source_file: DiscoveredFile) -> str:
    guessed, _ = mimetypes.guess_type(source_file.path)
    return guessed or "application/octet-stream"
