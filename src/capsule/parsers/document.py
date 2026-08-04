"""Unified, hierarchy-aware ingestion for text-bearing documents.

All supported document formats are converted to Markdown first.  The Markdown
structural splitter is then intentionally used twice: a larger pass produces
non-indexed parent context and a smaller pass produces the children that are
sent to enrichment and vector indexing.
"""

import asyncio
import re
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from capsule.config import (
    DOCUMENT_CHUNK_MAX_TOKENS,
    DOCUMENT_CHUNK_MERGE_MAX_TOKENS,
    DOCUMENT_CHUNK_MIN_TOKENS,
    DOCUMENT_CHUNK_TARGET_TOKENS,
)
from capsule.enums import AssetIndexRole, AssetType
from capsule.model_clients.tokenization import TokenCounter
from capsule.parsers.document_media import (
    DocumentMediaExtractor,
    MediaExtraction,
    validate_docx_package,
)
from capsule.parsers.markdown import MarkdownAssetBlock, MarkdownParser
from capsule.schemas import AssetDraft, SourceContext

DOCUMENT_PARENT_MAX_TOKENS = 2000
_NATIVE_MARKDOWN_EXTENSIONS = frozenset({".md", ".markdown"})
_MARKDOWN_DATA_IMAGE = re.compile(
    r"!\[([^\]]*)\]\(\s*data:image/[^)]*\)",
    re.IGNORECASE,
)
_HTML_DATA_IMAGE = re.compile(
    r"<img\b[^>]*\bsrc=(?:\"data:image/[^\"]*\"|'data:image/[^']*')[^>]*>",
    re.IGNORECASE,
)


class MarkItDownResult(Protocol):
    """The small portion of MarkItDown's conversion result used here."""

    text_content: str


class MarkItDownConverter(Protocol):
    def convert(self, source: Path) -> MarkItDownResult: ...


class DocumentParser:
    """Convert a document to Markdown and emit parent + text-child drafts.

    ``converter_factory`` exists for dependency injection. Native Markdown is
    read without rewriting; TXT, DOCX and PDF are adapted through MarkItDown.
    """

    def __init__(
        self,
        *,
        markdown_parser: MarkdownParser | None = None,
        converter_factory: Callable[[], MarkItDownConverter] | None = None,
        media_extractor: DocumentMediaExtractor | None = None,
    ) -> None:
        self._markdown = markdown_parser or MarkdownParser()
        self._converter_factory = converter_factory or _make_markitdown_converter
        self._media_extractor = media_extractor

    async def assetize_file(
        self,
        path: Path,
        token_counter: TokenCounter,
        *,
        min_tokens: int = DOCUMENT_CHUNK_MIN_TOKENS,
        target_tokens: int = DOCUMENT_CHUNK_TARGET_TOKENS,
        max_tokens: int = DOCUMENT_CHUNK_MAX_TOKENS,
        merge_max_tokens: int = DOCUMENT_CHUNK_MERGE_MAX_TOKENS,
        parent_max_tokens: int = DOCUMENT_PARENT_MAX_TOKENS,
    ) -> list[AssetDraft]:
        """Read/convert a file and return hierarchy-aware document drafts."""
        suffix = path.suffix.lower()
        if suffix == ".docx":
            await asyncio.to_thread(validate_docx_package, path)
        if suffix in _NATIVE_MARKDOWN_EXTENSIONS:
            markdown = await asyncio.to_thread(path.read_text, encoding="utf-8")
            converted_by = "native_markdown"
        else:
            markdown = await asyncio.to_thread(self._convert_file, path)
            converted_by = "markitdown"
            if suffix == ".docx":
                markdown = _remove_embedded_image_payloads(markdown)
        drafts = await self.assetize_markdown(
            markdown,
            path.name,
            token_counter,
            document_format=suffix.removeprefix(".") or "unknown",
            converted_by=converted_by,
            min_tokens=min_tokens,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            merge_max_tokens=merge_max_tokens,
            parent_max_tokens=parent_max_tokens,
        )
        if self._media_extractor is None or suffix not in {".docx", ".pdf"}:
            return drafts
        extraction = await asyncio.to_thread(self._media_extractor.extract, path)
        return [
            *drafts,
            *(await _embedded_media_drafts(path.name, suffix, extraction, token_counter)),
        ]

    async def assetize_markdown(
        self,
        markdown: str,
        file_name: str,
        token_counter: TokenCounter,
        *,
        document_format: str = "markdown",
        converted_by: str = "native_markdown",
        min_tokens: int = DOCUMENT_CHUNK_MIN_TOKENS,
        target_tokens: int = DOCUMENT_CHUNK_TARGET_TOKENS,
        max_tokens: int = DOCUMENT_CHUNK_MAX_TOKENS,
        merge_max_tokens: int = DOCUMENT_CHUNK_MERGE_MAX_TOKENS,
        parent_max_tokens: int = DOCUMENT_PARENT_MAX_TOKENS,
    ) -> list[AssetDraft]:
        """Build parent and child drafts from already-converted Markdown."""
        if not 1 <= min_tokens <= target_tokens <= max_tokens <= merge_max_tokens:
            raise ValueError("token limits must satisfy min <= target <= max <= merge_max")
        if parent_max_tokens < merge_max_tokens:
            raise ValueError("parent_max_tokens must be greater than or equal to merge_max_tokens")

        parents = await self._markdown.split(
            markdown,
            token_counter,
            min_tokens=min_tokens,
            target_tokens=parent_max_tokens,
            max_tokens=parent_max_tokens,
            merge_max_tokens=parent_max_tokens,
        )
        drafts: list[AssetDraft] = []
        document_child_index = 0
        for parent_index, parent in enumerate(parents):
            hierarchy_key = f"document-parent-{parent_index}"
            parent_heading = list(parent.heading_path)
            parent_locator = _locator(
                document_format=document_format,
                parent_index=parent_index,
                block_index=None,
                heading_path=parent_heading,
                char_start=parent.char_start,
                char_end=parent.char_end,
                kind="document_parent",
            )
            drafts.append(
                AssetDraft(
                    asset_type=AssetType.MARKDOWN_BLOCK,
                    file_name=file_name,
                    index_role=AssetIndexRole.PARENT,
                    hierarchy_key=hierarchy_key,
                    source_locator=parent_locator,
                    source_contexts=_heading_context(parent_heading),
                    raw_content=parent.raw,
                    file_info=_file_info(
                        parent,
                        document_format=document_format,
                        converted_by=converted_by,
                        hierarchy_role="parent",
                    ),
                )
            )

            children = await self._markdown.split(
                parent.raw,
                token_counter,
                min_tokens=min_tokens,
                target_tokens=target_tokens,
                max_tokens=max_tokens,
                merge_max_tokens=merge_max_tokens,
            )
            for child_order, child in enumerate(children):
                heading_path = _child_heading_path(parent_heading, child.heading_path)
                char_start = parent.char_start + child.char_start
                char_end = parent.char_start + child.char_end
                drafts.append(
                    AssetDraft(
                        asset_type=AssetType.MARKDOWN_BLOCK,
                        file_name=file_name,
                        index_role=AssetIndexRole.CHILD,
                        parent_hierarchy_key=hierarchy_key,
                        child_order=child_order,
                        source_locator=_locator(
                            document_format=document_format,
                            parent_index=parent_index,
                            block_index=child_order,
                            document_child_index=document_child_index,
                            heading_path=heading_path,
                            char_start=char_start,
                            char_end=char_end,
                            kind="text_range",
                        ),
                        source_contexts=_heading_context(heading_path),
                        raw_content=child.raw,
                        file_info=_file_info(
                            child,
                            document_format=document_format,
                            converted_by=converted_by,
                            hierarchy_role="child",
                        ),
                    )
                )
                document_child_index += 1
        return drafts

    def _convert_file(self, path: Path) -> str:
        result = self._converter_factory().convert(path)
        text_content = result.text_content
        if not isinstance(text_content, str):
            raise TypeError("MarkItDown conversion result has no string text_content")
        return text_content


def _make_markitdown_converter() -> MarkItDownConverter:
    try:
        module = import_module("markitdown")
    except ImportError as exc:  # pragma: no cover - optional dependency absent
        raise ImportError(
            "MarkItDown is required to ingest this document type; install the markitdown package"
        ) from exc
    factory = cast(Callable[[], MarkItDownConverter], module.MarkItDown)
    return factory()


def _remove_embedded_image_payloads(markdown: str) -> str:
    """Keep useful alt text without indexing base64 image payloads as prose."""

    def replace_markdown(match: re.Match[str]) -> str:
        alt_text = match.group(1).strip()
        return f"[Embedded image: {alt_text}]" if alt_text else ""

    cleaned = _MARKDOWN_DATA_IMAGE.sub(replace_markdown, markdown)
    return _HTML_DATA_IMAGE.sub("", cleaned)


def _locator(
    *,
    document_format: str,
    parent_index: int,
    block_index: int | None,
    document_child_index: int | None = None,
    heading_path: list[str],
    char_start: int,
    char_end: int,
    kind: str,
) -> dict[str, object]:
    locator: dict[str, object] = {
        "type": kind,
        "document_format": document_format,
        "parent_block_index": parent_index,
        "heading_path": heading_path,
        "char_start": char_start,
        "char_end": char_end,
    }
    if block_index is not None:
        locator["block_index"] = block_index
    if document_child_index is not None:
        locator["document_child_index"] = document_child_index
    return locator


def _file_info(
    block: MarkdownAssetBlock,
    *,
    document_format: str,
    converted_by: str,
    hierarchy_role: str,
) -> dict[str, object]:
    return {
        "token_count": block.token_count,
        "oversized": block.oversized,
        "oversized_reason": block.oversized_reason,
        "document_format": document_format,
        "converted_by": converted_by,
        "hierarchy_role": hierarchy_role,
    }


def _heading_context(heading_path: list[str]) -> list[SourceContext]:
    if not heading_path:
        return []
    return [
        SourceContext(
            text=" > ".join(heading_path),
            relation_type="heading_context",
            heading_path=heading_path,
        )
    ]


def _child_heading_path(parent: list[str], child: list[str]) -> list[str]:
    if not child:
        return list(parent)
    if child[: len(parent)] == parent:
        return list(child)
    return [*parent, *child]


# Explicit alias makes the conversion dependency discoverable to callers while
# retaining the neutral ``DocumentParser`` name for the common ingestion path.
MarkItDownDocumentParser = DocumentParser


async def _embedded_media_drafts(
    file_name: str,
    suffix: str,
    extraction: MediaExtraction,
    token_counter: TokenCounter,
) -> list[AssetDraft]:
    """Represent extracted media as image children of one factual media parent.

    MarkItDown does not expose stable text offsets for every DOCX/PDF image. A
    dedicated media parent therefore avoids inventing a relationship with an
    unrelated text chunk while still preserving the document-level hierarchy.
    """

    if not extraction.images:
        return []
    hierarchy_key = "document-media-parent"
    ocr_image_count = sum(image.ocr is not None for image in extraction.images)
    parent_content = (
        "# Embedded document media\n\n"
        f"Extracted images: {len(extraction.images)}; images with OCR text: {ocr_image_count}."
    )
    parent_token_count = (await token_counter.count_many([parent_content]))[0]
    parent = AssetDraft(
        asset_type=AssetType.MARKDOWN_BLOCK,
        file_name=file_name,
        index_role=AssetIndexRole.PARENT,
        hierarchy_key=hierarchy_key,
        source_locator={
            "type": "document_media_collection",
            "document_format": suffix.removeprefix("."),
            "image_count": len(extraction.images),
        },
        raw_content=parent_content,
        file_info={
            "token_count": parent_token_count,
            "document_format": suffix.removeprefix("."),
            "hierarchy_role": "parent",
            "embedded_image_count": len(extraction.images),
            "skipped_image_count": len(extraction.skipped),
        },
    )
    children: list[AssetDraft] = []
    for image_index, image in enumerate(extraction.images):
        locator = {
            **image.source_locator,
            "image_index": image_index,
        }
        contexts: list[SourceContext] = []
        raw_content: str | None = None
        file_info = {
            **image.file_info,
            "document_format": suffix.removeprefix("."),
            "hierarchy_role": "child",
            "content_hash": image.content_hash,
        }
        if image.ocr is not None:
            raw_content = image.ocr.text
            ocr_info = file_info.get("ocr")
            if isinstance(ocr_info, dict):
                file_info["ocr"] = {**ocr_info, "text": image.ocr.text}
            contexts.append(
                SourceContext(
                    text=image.ocr.text,
                    relation_type="ocr_text",
                    source_path=file_name,
                )
            )
        children.append(
            AssetDraft(
                asset_type=AssetType.IMAGE,
                file_name=file_name,
                index_role=AssetIndexRole.CHILD,
                parent_hierarchy_key=hierarchy_key,
                child_order=image_index,
                source_locator=locator,
                source_contexts=contexts,
                raw_content=raw_content,
                file_info=file_info,
                derived_file_uri=image.derived_file_uri,
            )
        )
    return [parent, *children]
