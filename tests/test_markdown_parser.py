from collections.abc import Sequence

import pytest

from capsule.parsers.markdown import MarkdownParser


class CharacterTokenCounter:
    async def count_many(self, texts: Sequence[str]) -> list[int]:
        return [len(text) for text in texts]


def test_images_in_group_inherit_preceding_heading() -> None:
    source = """
# 午后-黄昏

![](images/one.png)
![](images/two.png)
"""

    result = MarkdownParser().parse(source)

    assert len(result.image_references) == 2
    for reference in result.image_references:
        assert reference.contexts[0].text == "午后-黄昏"
        assert reference.contexts[0].relation_type == "preceding_heading"


def test_image_caption_precedes_document_context() -> None:
    source = """
## 角色设定

![银发角色](images/character.png)
"""

    result = MarkdownParser().parse(source)
    contexts = result.image_references[0].contexts

    assert [context.relation_type for context in contexts] == [
        "caption",
        "preceding_heading",
    ]
    assert [context.text for context in contexts] == ["银发角色", "角色设定"]


@pytest.mark.asyncio
async def test_long_heading_section_recurses_to_child_heading() -> None:
    source = f"# 第一章\n\n{'甲' * 700}\n\n## 第二节\n\n{'乙' * 700}\n"

    assets = await MarkdownParser().assetize(
        source,
        "notes.md",
        CharacterTokenCounter(),
    )

    assert len(assets) == 2
    assert assets[0].source_locator["heading_path"] == ["第一章"]
    assert assets[1].source_locator["heading_path"] == ["第一章", "第二节"]
    for asset in assets:
        start = asset.source_locator["char_start"]
        end = asset.source_locator["char_end"]
        assert asset.raw_content == source[start:end]
        assert asset.file_info["token_count"] <= 500 or asset.file_info["oversized"]


@pytest.mark.asyncio
async def test_oversized_code_fence_is_not_split() -> None:
    code = "x" * 1300
    source = f"# Code\n\n```python\n{code}\n```\n"

    assets = await MarkdownParser().assetize(
        source,
        "code.md",
        CharacterTokenCounter(),
    )

    assert len(assets) == 1
    assert assets[0].raw_content == f"```python\n{code}\n```\n"
    assert assets[0].file_info["oversized"] is True
    assert assets[0].file_info["oversized_reason"] == "indivisible_code_fence"


@pytest.mark.asyncio
async def test_long_list_splits_only_between_top_level_items() -> None:
    items = [f"- {letter * 150}\n" for letter in ("甲", "乙", "丙", "丁")]
    source = "# List\n\n" + "".join(items)

    assets = await MarkdownParser().assetize(
        source,
        "list.md",
        CharacterTokenCounter(),
    )

    assert len(assets) == 2
    assert assets[0].raw_content == "".join(items[:2])
    assert assets[1].raw_content == "".join(items[2:])
    assert all(asset.raw_content.startswith("- ") for asset in assets)


@pytest.mark.asyncio
async def test_adjacent_short_child_sections_merge_under_same_parent() -> None:
    source = f"# Root\n\n{'长' * 1180}\n\n## A\n\na\n\n## B\n\nb\n"

    assets = await MarkdownParser().assetize(
        source,
        "short.md",
        CharacterTokenCounter(),
    )

    assert len(assets) == 2
    assert assets[1].source_locator["heading_path"] == ["Root"]
    assert "## A" in assets[1].raw_content
    assert "## B" in assets[1].raw_content


@pytest.mark.asyncio
async def test_short_block_merge_may_use_600_token_ceiling() -> None:
    source = f"# Root\n\n## A\n\n{'甲' * 370}\n\n## B\n\n{'乙' * 170}\n"

    assets = await MarkdownParser().assetize(
        source,
        "short-merge.md",
        CharacterTokenCounter(),
    )

    assert len(assets) == 1
    assert 500 < assets[0].file_info["token_count"] <= 600
    assert "## A" in assets[0].raw_content
    assert "## B" in assets[0].raw_content


@pytest.mark.asyncio
async def test_long_paragraph_splits_on_sentence_boundaries() -> None:
    sentences = [f"{letter * 180}。\n" for letter in ("甲", "乙", "丙", "丁")]
    source = "# 正文\n\n" + "".join(sentences)

    assets = await MarkdownParser().assetize(
        source,
        "paragraph.md",
        CharacterTokenCounter(),
    )

    assert len(assets) == 2
    assert assets[0].raw_content == "".join(sentences[:2])
    assert assets[1].raw_content == "".join(sentences[2:])
    assert all(250 <= asset.file_info["token_count"] <= 500 for asset in assets)


@pytest.mark.asyncio
async def test_oversized_table_is_never_split() -> None:
    rows = [f"| 第{index}行 | {'内容' * 40} |\n" for index in range(12)]
    table = "| 行 | 内容 |\n| --- | --- |\n" + "".join(rows)
    source = "# 表格\n\n" + table

    assets = await MarkdownParser().assetize(
        source,
        "table.md",
        CharacterTokenCounter(),
    )

    assert len(assets) == 1
    assert assets[0].raw_content == table
    assert assets[0].file_info["token_count"] > 600
    assert assets[0].file_info["oversized"] is True
    assert assets[0].file_info["oversized_reason"] == "indivisible_table"
