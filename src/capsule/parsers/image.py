"""Whole-file image assetization and factual metadata extraction."""

import asyncio
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image

from capsule.enums import AssetType
from capsule.schemas import AssetDraft, DiscoveredFile


class ImageParser:
    async def assetize(self, source_file: DiscoveredFile) -> list[AssetDraft]:
        path = Path(source_file.path)
        file_info = await asyncio.to_thread(
            extract_image_info,
            path,
            source_file.relative_path,
        )
        return [
            AssetDraft(
                asset_type=AssetType.IMAGE,
                file_name=path.name,
                source_locator={"type": "whole_file"},
                raw_content=None,
                file_info=file_info,
                preview_uri=None,
            )
        ]


def extract_image_info(path: Path, relative_path: str | None = None) -> dict[str, Any]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        raw_exif = image.getexif()
        exif = {
            ExifTags.TAGS.get(tag, str(tag)): _json_safe(value)
            for tag, value in raw_exif.items()
        }
        width, height = image.size
        image_format = image.format
        mime_type = Image.MIME.get(image_format or "")
        folder_context = _folder_context(relative_path)
        return {
            "width": width,
            "height": height,
            "aspect_ratio": width / height if height else None,
            "mime_type": mime_type,
            "file_size_bytes": path.stat().st_size,
            "format": image_format,
            "color_mode": image.mode,
            "exif": exif,
            "captured_at": _first_exif_value(
                exif,
                "DateTimeOriginal",
                "DateTimeDigitized",
                "DateTime",
            ),
            "software": _first_exif_value(exif, "Software"),
            "folder_context": folder_context,
        }


def _folder_context(relative_path: str | None) -> list[str]:
    if not relative_path:
        return []
    parent = Path(relative_path).parent
    return [] if parent == Path(".") else list(parent.parts)


def _first_exif_value(exif: dict[str, Any], *keys: str) -> Any | None:
    for key in keys:
        value = exif.get(key)
        if value not in (None, ""):
            return value
    return None


def _json_safe(value: object) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)
