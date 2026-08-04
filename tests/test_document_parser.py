import io
import zipfile
from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image

from capsule.enums import AssetIndexRole, AssetType
from capsule.parsers.document import DocumentParser
from capsule.parsers.document_media import DocumentMediaExtractor


class CharacterTokenCounter:
    async def count_many(self, texts: Sequence[str]) -> list[int]:
        return [len(text) for text in texts]


class FakeConversionResult:
    def __init__(self, text_content: str) -> None:
        self.text_content = text_content


class FakeMarkItDown:
    def __init__(self, output: str) -> None:
        self._output = output
        self.paths: list[Path] = []

    def convert(self, source: Path) -> FakeConversionResult:
        self.paths.append(source)
        return FakeConversionResult(self._output)


@pytest.mark.asyncio
async def test_document_parser_builds_bounded_parents_and_indexable_children() -> None:
    first_section = ("甲" * 299 + "。\n") * 3
    second_section = ("乙" * 299 + "。\n") * 3
    source = f"# 第一章\n\n{first_section}\n\n# 第二章\n\n{second_section}\n"

    drafts = await DocumentParser().assetize_markdown(
        source,
        "notes.docx",
        CharacterTokenCounter(),
        document_format="docx",
        converted_by="markitdown",
        max_tokens=400,
        parent_max_tokens=1000,
    )

    parents = [draft for draft in drafts if draft.index_role is AssetIndexRole.PARENT]
    children = [draft for draft in drafts if draft.index_role is AssetIndexRole.CHILD]

    assert len(parents) == 2
    assert all(parent.file_info["token_count"] <= 1000 for parent in parents)
    assert all(parent.asset_type is AssetType.MARKDOWN_BLOCK for parent in parents)
    children_by_parent = {
        parent.hierarchy_key: [
            child.child_order
            for child in children
            if child.parent_hierarchy_key == parent.hierarchy_key
        ]
        for parent in parents
    }
    assert all(orders == list(range(len(orders))) for orders in children_by_parent.values())
    assert {child.parent_hierarchy_key for child in children} == {
        parent.hierarchy_key for parent in parents
    }
    assert all(child.file_info["token_count"] <= 400 for child in children)
    assert all(child.source_locator["document_format"] == "docx" for child in children)


@pytest.mark.asyncio
async def test_children_keep_global_offsets_and_heading_context() -> None:
    source = f"# Root\n\n## Section\n\n{'甲' * 900}\n"

    drafts = await DocumentParser().assetize_markdown(
        source,
        "notes.md",
        CharacterTokenCounter(),
        max_tokens=400,
        parent_max_tokens=1600,
    )

    children = [draft for draft in drafts if draft.index_role is AssetIndexRole.CHILD]
    assert children
    for child in children:
        start = child.source_locator["char_start"]
        end = child.source_locator["char_end"]
        assert child.raw_content == source[start:end]
        assert child.source_contexts[0].heading_path == ["Root", "Section"]


@pytest.mark.asyncio
async def test_docx_is_converted_through_injected_markitdown_adapter(tmp_path: Path) -> None:
    source_path = tmp_path / "report.docx"
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr("word/document.xml", "<document />")
    fake = FakeMarkItDown("# Converted\n\n正文。\n")
    parser = DocumentParser(converter_factory=lambda: fake)

    drafts = await parser.assetize_file(source_path, CharacterTokenCounter())

    assert fake.paths == [source_path]
    assert [draft.index_role for draft in drafts] == [
        AssetIndexRole.PARENT,
        AssetIndexRole.CHILD,
    ]
    assert all(draft.file_info["converted_by"] == "markitdown" for draft in drafts)
    assert drafts[1].raw_content == "# Converted\n\n正文。\n"


@pytest.mark.asyncio
async def test_docx_data_uri_image_payload_is_not_indexed_as_text(tmp_path: Path) -> None:
    source_path = tmp_path / "report.docx"
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr("word/document.xml", "<document />")
    fake = FakeMarkItDown(
        "# Report\n\n![Invoice scan](data:image/png;base64,AAAAABBBBBCCCC)\n"
    )

    drafts = await DocumentParser(converter_factory=lambda: fake).assetize_file(
        source_path,
        CharacterTokenCounter(),
    )

    indexed_text = "\n".join(draft.raw_content or "" for draft in drafts)
    assert "Invoice scan" in indexed_text
    assert "base64" not in indexed_text
    assert "AAAAABBBBBCCCC" not in indexed_text


@pytest.mark.asyncio
async def test_plain_text_is_adapted_by_markitdown_to_the_same_hierarchy(tmp_path: Path) -> None:
    source_path = tmp_path / "notes.txt"
    source_path.write_text("标题\n\n第一段。\n\n第二段。\n", encoding="utf-8")
    fake = FakeMarkItDown("标题\n\n第一段。\n\n第二段。\n")

    drafts = await DocumentParser(converter_factory=lambda: fake).assetize_file(
        source_path,
        CharacterTokenCounter(),
    )

    assert [draft.index_role for draft in drafts] == [
        AssetIndexRole.PARENT,
        AssetIndexRole.CHILD,
    ]
    assert fake.paths == [source_path]
    assert {draft.file_info["converted_by"] for draft in drafts} == {"markitdown"}


@pytest.mark.asyncio
async def test_real_markitdown_plain_text_smoke(tmp_path: Path) -> None:
    source_path = tmp_path / "notes.txt"
    source_path.write_text("标题\n\n正文。\n", encoding="utf-8")

    drafts = await DocumentParser().assetize_file(source_path, CharacterTokenCounter())

    assert [draft.index_role for draft in drafts] == [
        AssetIndexRole.PARENT,
        AssetIndexRole.CHILD,
    ]
    assert drafts[1].raw_content == "标题\n\n正文。\n"


@pytest.mark.asyncio
async def test_parent_limit_must_not_be_smaller_than_child_limit() -> None:
    with pytest.raises(ValueError, match="parent_max_tokens"):
        await DocumentParser().assetize_markdown(
            "正文。",
            "notes.md",
            CharacterTokenCounter(),
            max_tokens=400,
            parent_max_tokens=399,
        )


@pytest.mark.asyncio
async def test_docx_embedded_images_become_ocr_enriched_image_children(tmp_path: Path) -> None:
    source_path = tmp_path / "report.docx"
    image_bytes = io.BytesIO()
    Image.new("RGB", (80, 80), (20, 100, 180)).save(image_bytes, format="PNG")
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="urn:w" xmlns:a="urn:a" xmlns:r="urn:r">'
            '<a:blip r:embed="rId1"/></w:document>',
        )
        archive.writestr(
            "word/_rels/document.xml.rels",
            '<Relationships><Relationship Id="rId1" Target="media/image.png"/>'
            "</Relationships>",
        )
        archive.writestr("word/media/image.png", image_bytes.getvalue())

    parser = DocumentParser(
        converter_factory=lambda: FakeMarkItDown("# Report\n\n正文。\n"),
        media_extractor=DocumentMediaExtractor(
            tmp_path / "media",
            ocr_engine=lambda _: [[[], "图内标题", 0.96]],
        ),
    )
    drafts = await parser.assetize_file(source_path, CharacterTokenCounter())

    image_draft = next(draft for draft in drafts if draft.asset_type is AssetType.IMAGE)
    media_parent = next(
        draft
        for draft in drafts
        if draft.source_locator.get("type") == "document_media_collection"
    )
    assert image_draft.index_role is AssetIndexRole.CHILD
    assert image_draft.parent_hierarchy_key == media_parent.hierarchy_key
    assert image_draft.derived_file_uri is not None
    assert image_draft.raw_content == "图内标题"
    assert image_draft.source_contexts[0].relation_type == "ocr_text"
    assert image_draft.file_info["ocr"]["text"] == "图内标题"
