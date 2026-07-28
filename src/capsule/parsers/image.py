from pathlib import Path
from typing import Any

from PIL import ExifTags, Image


def extract_image_info(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        exif = {
            ExifTags.TAGS.get(tag, str(tag)): _json_safe(value)
            for tag, value in image.getexif().items()
        }
        width, height = image.size
        return {
            "width": width,
            "height": height,
            "aspect_ratio": width / height if height else None,
            "format": image.format,
            "color_mode": image.mode,
            "exif": exif,
        }


def _json_safe(value: object) -> str | int | float | bool | None:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)
