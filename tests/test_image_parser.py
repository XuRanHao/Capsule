from pathlib import Path

import pytest
from PIL import Image

from capsule.enums import AssetType
from capsule.parsers.image import ImageParser, extract_image_info
from capsule.schemas import DiscoveredFile


def test_extract_image_info_uses_factual_file_values(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    Image.new("RGB", (320, 180), (10, 20, 30)).save(path)

    info = extract_image_info(path, "project/images/sample.png")

    assert info["width"] == 320
    assert info["height"] == 180
    assert info["aspect_ratio"] == pytest.approx(16 / 9)
    assert info["mime_type"] == "image/png"
    assert info["file_size_bytes"] == path.stat().st_size
    assert info["color_mode"] == "RGB"
    assert info["exif"] == {}
    assert info["captured_at"] is None
    assert info["software"] is None
    assert info["folder_context"] == ["project", "images"]


@pytest.mark.asyncio
async def test_image_parser_outputs_one_whole_file_asset(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    Image.new("RGBA", (10, 20), (0, 0, 0, 0)).save(path)
    source = DiscoveredFile(
        path=str(path),
        relative_path="sample.png",
        extension=".png",
        size_bytes=path.stat().st_size,
    )

    assets = await ImageParser().assetize(source)

    assert len(assets) == 1
    assert assets[0].asset_type is AssetType.IMAGE
    assert assets[0].source_locator == {"type": "whole_file"}
    assert assets[0].raw_content is None
    assert assets[0].preview_uri is None
