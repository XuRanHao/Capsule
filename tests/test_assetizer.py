import logging

import pytest

from capsule.enums import AssetType
from capsule.parsers.assetizer import Assetizer
from capsule.schemas import AssetDraft, DiscoveredFile


def discovered_file(extension: str = ".md") -> DiscoveredFile:
    return DiscoveredFile(
        path="/input/example.md",
        relative_path="example.md",
        extension=extension,
        size_bytes=42,
    )


@pytest.mark.asyncio
async def test_assetize_dispatches_to_registered_handler() -> None:
    async def markdown_handler(source_file: DiscoveredFile) -> list[AssetDraft]:
        return [
            AssetDraft(
                asset_type=AssetType.MARKDOWN_BLOCK,
                file_name=source_file.relative_path,
                raw_content="# Capsule",
            )
        ]

    result = await Assetizer({"MD": markdown_handler}).assetize(discovered_file())

    assert result.succeeded is True
    assert result.error_message is None
    assert len(result.assets) == 1
    assert result.assets[0].asset_type is AssetType.MARKDOWN_BLOCK


@pytest.mark.asyncio
async def test_assetize_records_handler_failure_without_raising(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def broken_handler(_source_file: DiscoveredFile) -> list[AssetDraft]:
        raise ValueError("invalid markdown")

    with caplog.at_level(logging.ERROR, logger="capsule.parsers.assetizer"):
        result = await Assetizer({".md": broken_handler}).assetize(discovered_file())

    assert result.succeeded is False
    assert result.assets == []
    assert result.error_message == "invalid markdown"
    assert "assetization failed for example.md" in caplog.text


@pytest.mark.asyncio
async def test_assetize_reports_unregistered_extension(
    caplog: pytest.LogCaptureFixture,
) -> None:
    source_file = discovered_file(".txt")

    with caplog.at_level(logging.WARNING, logger="capsule.parsers.assetizer"):
        result = await Assetizer({}).assetize(source_file)

    assert result.succeeded is False
    assert result.assets == []
    assert result.error_message == "no asset handler registered for extension: .txt"
    assert "assetization skipped" in caplog.text
