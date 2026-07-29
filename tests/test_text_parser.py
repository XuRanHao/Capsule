from collections.abc import Sequence

import pytest

from capsule.enums import AssetType
from capsule.parsers.text import TextParser


class CharacterTokenCounter:
    async def count_many(self, texts: Sequence[str]) -> list[int]:
        return [len(text) for text in texts]


@pytest.mark.asyncio
async def test_hard_wrapped_paragraphs_keep_original_offsets_and_merge_when_short() -> None:
    source = "第一行只是自动换行\n第二行仍属于同一段。\n\n相邻短段。\n"

    assets = await TextParser().assetize(source, "notes.txt", CharacterTokenCounter())

    assert len(assets) == 1
    asset = assets[0]
    assert asset.asset_type is AssetType.TEXT_BLOCK
    assert asset.raw_content == source
    assert "heading_path" not in asset.source_locator
    assert asset.raw_content == source[
        asset.source_locator["char_start"] : asset.source_locator["char_end"]
    ]
    assert asset.file_info["node_kinds"] == ["paragraph"]


@pytest.mark.asyncio
async def test_long_paragraph_splits_only_at_sentence_boundaries() -> None:
    sentence = "甲" * 150 + "。"
    source = sentence * 3

    assets = await TextParser().assetize(source, "long.txt", CharacterTokenCounter())

    assert [asset.raw_content for asset in assets] == [sentence * 2, sentence]
    assert all(asset.file_info["token_count"] <= 400 for asset in assets)
    assert all(asset.raw_content.endswith("。") for asset in assets)


@pytest.mark.asyncio
async def test_long_list_splits_between_top_level_items() -> None:
    items = [f"- {letter * 150}\n" for letter in ("甲", "乙", "丙")]
    source = "".join(items)

    assets = await TextParser().assetize(source, "list.txt", CharacterTokenCounter())

    assert [asset.raw_content for asset in assets] == ["".join(items[:2]), items[2]]
    assert all(asset.raw_content.startswith("- ") for asset in assets)
    assert all(asset.file_info["node_kinds"] == ["list"] for asset in assets)


@pytest.mark.asyncio
async def test_oversized_fenced_code_is_not_split() -> None:
    source = "```python\n" + "x" * 1300 + "\n```\n"

    assets = await TextParser().assetize(source, "code.txt", CharacterTokenCounter())

    assert len(assets) == 1
    assert assets[0].raw_content == source
    assert assets[0].file_info["oversized"] is True
    assert assets[0].file_info["oversized_reason"] == "indivisible_code_fence"
