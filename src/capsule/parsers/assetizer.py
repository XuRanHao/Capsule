"""Unified entry point for converting discovered files into asset drafts."""

import logging
from collections.abc import Awaitable, Callable, Mapping

from pydantic import BaseModel, Field

from capsule.schemas import AssetDraft, DiscoveredFile

AssetHandler = Callable[[DiscoveredFile], Awaitable[list[AssetDraft]]]

logger = logging.getLogger(__name__)


class AssetizationResult(BaseModel):
    """Outcome for one file; processing failures are returned instead of raised."""

    source_file: DiscoveredFile
    succeeded: bool
    assets: list[AssetDraft] = Field(default_factory=list)
    error_message: str | None = None


class Assetizer:
    """Dispatch a discovered file to a registered format-specific handler."""

    def __init__(self, handlers: Mapping[str, AssetHandler]) -> None:
        self._handlers = {
            _normalize_extension(extension): handler
            for extension, handler in handlers.items()
        }

    async def assetize(self, source_file: DiscoveredFile) -> AssetizationResult:
        extension = _normalize_extension(source_file.extension)
        handler = self._handlers.get(extension)
        if handler is None:
            message = f"no asset handler registered for extension: {extension}"
            logger.warning(
                "assetization skipped: %s",
                message,
                extra={
                    "source_path": source_file.path,
                    "relative_path": source_file.relative_path,
                    "source_extension": extension,
                },
            )
            return AssetizationResult(
                source_file=source_file,
                succeeded=False,
                error_message=message,
            )

        try:
            assets = await handler(source_file)
        except Exception as exc:
            logger.exception(
                "assetization failed for %s",
                source_file.relative_path,
                extra={
                    "source_path": source_file.path,
                    "relative_path": source_file.relative_path,
                    "source_extension": extension,
                },
            )
            return AssetizationResult(
                source_file=source_file,
                succeeded=False,
                error_message=str(exc),
            )

        return AssetizationResult(
            source_file=source_file,
            succeeded=True,
            assets=assets,
        )


def _normalize_extension(extension: str) -> str:
    normalized = extension.strip().lower()
    return normalized if normalized.startswith(".") else f".{normalized}"
