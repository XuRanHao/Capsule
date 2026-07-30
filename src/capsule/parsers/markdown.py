import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token

from capsule.config import DOCUMENT_CHUNK_MAX_TOKENS
from capsule.enums import AssetType
from capsule.model_clients.tokenization import TokenCounter
from capsule.schemas import AssetDraft, SourceContext


@dataclass(slots=True)
class MarkdownTextBlock:
    block_index: int
    text: str
    heading_path: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MarkdownImageReference:
    image_path: str
    contexts: list[SourceContext] = field(default_factory=list)
    heading_path: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MarkdownParseResult:
    text_blocks: list[MarkdownTextBlock] = field(default_factory=list)
    image_references: list[MarkdownImageReference] = field(default_factory=list)


@dataclass(slots=True)
class MarkdownNode:
    kind: str
    char_start: int
    char_end: int
    raw: str
    heading_path: list[str] = field(default_factory=list)
    heading_level: int | None = None
    token_count: int = 0


@dataclass(slots=True)
class MarkdownAssetBlock:
    char_start: int
    char_end: int
    heading_path: list[str]
    raw: str
    token_count: int
    oversized: bool = False
    oversized_reason: str | None = None


class MarkdownParser:
    """Extract text blocks and source context for referenced images."""

    def __init__(self) -> None:
        self._parser = MarkdownIt("commonmark").enable("table")

    def parse_file(self, path: Path) -> MarkdownParseResult:
        return self.parse(path.read_text(encoding="utf-8"))

    def parse(self, source: str) -> MarkdownParseResult:
        tokens = self._parser.parse(source)
        result = MarkdownParseResult()
        heading_path: list[str] = []
        last_text: tuple[str, str, int] | None = None
        block_index = 0

        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token.type in {"heading_open", "paragraph_open"}:
                inline = _next_inline(tokens, index)
                if inline is None:
                    index += 1
                    continue

                text = _inline_text(inline).strip()
                images = _inline_images(inline)
                relation_type = (
                    "preceding_heading" if token.type == "heading_open" else "preceding_text"
                )

                if token.type == "heading_open" and text:
                    level = int(token.tag.removeprefix("h"))
                    heading_path = heading_path[: level - 1]
                    heading_path.append(text)

                if text:
                    result.text_blocks.append(
                        MarkdownTextBlock(
                            block_index=block_index,
                            text=text,
                            heading_path=list(heading_path),
                        )
                    )
                    last_text = (text, relation_type, block_index)
                    block_index += 1

                for image in images:
                    contexts: list[SourceContext] = []
                    alt_text = image.content.strip()
                    if alt_text:
                        contexts.append(
                            SourceContext(
                                text=alt_text,
                                relation_type="caption",
                                text_block_index=None,
                            )
                        )
                    if last_text is not None:
                        context_text, context_type, context_index = last_text
                        if not alt_text or alt_text != context_text:
                            contexts.append(
                                SourceContext(
                                    text=context_text,
                                    relation_type=context_type,
                                    text_block_index=context_index,
                                )
                            )
                    result.image_references.append(
                        MarkdownImageReference(
                            image_path=str(image.attrGet("src") or ""),
                            contexts=contexts,
                            heading_path=list(heading_path),
                        )
                    )
            index += 1

        return result

    async def assetize_file(
        self,
        path: Path,
        token_counter: TokenCounter,
        *,
        max_tokens: int = DOCUMENT_CHUNK_MAX_TOKENS,
    ) -> list[AssetDraft]:
        source = await asyncio.to_thread(_read_markdown, path)
        return await self.assetize(source, path.name, token_counter, max_tokens=max_tokens)

    async def assetize(
        self,
        source: str,
        file_name: str,
        token_counter: TokenCounter,
        *,
        max_tokens: int = DOCUMENT_CHUNK_MAX_TOKENS,
    ) -> list[AssetDraft]:
        nodes = self._structured_nodes(source)
        if not _has_content(nodes):
            return []
        counts = await token_counter.count_many([node.raw for node in nodes])
        for node, count in zip(nodes, counts, strict=True):
            node.token_count = count
        blocks = await _split_by_heading(
            source,
            nodes,
            token_counter,
            max_tokens=max_tokens,
            heading_level=1,
        )
        blocks = await _merge_short_blocks(source, blocks, token_counter, max_tokens=max_tokens)
        return [
            AssetDraft(
                asset_type=AssetType.MARKDOWN_BLOCK,
                file_name=file_name,
                source_locator={
                    "type": "text_range",
                    "block_index": index,
                    "heading_path": block.heading_path,
                    "char_start": block.char_start,
                    "char_end": block.char_end,
                },
                raw_content=block.raw,
                file_info={
                    "token_count": block.token_count,
                    "oversized": block.oversized,
                    "oversized_reason": block.oversized_reason,
                },
            )
            for index, block in enumerate(blocks)
        ]

    def _structured_nodes(self, source: str) -> list[MarkdownNode]:
        tokens = self._parser.parse(source)
        offsets = _line_offsets(source)
        nodes: list[MarkdownNode] = []
        heading_path: list[str] = []
        recognized = {
            "paragraph_open": "paragraph",
            "bullet_list_open": "list",
            "ordered_list_open": "list",
            "blockquote_open": "blockquote",
            "table_open": "table",
            "fence": "code_fence",
            "code_block": "code_fence",
            "hr": "horizontal_rule",
        }
        for index, token in enumerate(tokens):
            if token.level != 0 or token.map is None:
                continue
            kind = recognized.get(token.type)
            heading_level: int | None = None
            if token.type == "heading_open":
                kind = "heading"
                heading_level = int(token.tag.removeprefix("h"))
                inline = _next_inline(tokens, index)
                heading_text = _inline_text(inline).strip() if inline else ""
                heading_path = heading_path[: heading_level - 1]
                heading_path.append(heading_text)
            if kind is None:
                continue
            start_line, end_line = token.map
            char_start = offsets[start_line]
            char_end = offsets[end_line]
            nodes.append(
                MarkdownNode(
                    kind=kind,
                    char_start=char_start,
                    char_end=char_end,
                    raw=source[char_start:char_end],
                    heading_path=list(heading_path),
                    heading_level=heading_level,
                )
            )
        return nodes


def _next_inline(tokens: list[Token], start: int) -> Token | None:
    for token in tokens[start + 1 :]:
        if token.type == "inline":
            return token
        if token.nesting == -1:
            return None
    return None


def _inline_text(token: Token) -> str:
    if not token.children:
        return token.content
    return "".join(child.content for child in token.children if child.type == "text")


def _inline_images(token: Token) -> list[Token]:
    return [child for child in token.children or [] if child.type == "image"]


async def _split_by_heading(
    source: str,
    nodes: list[MarkdownNode],
    counter: TokenCounter,
    *,
    max_tokens: int,
    heading_level: int,
) -> list[MarkdownAssetBlock]:
    if not _has_content(nodes):
        return []
    raw = source[nodes[0].char_start : nodes[-1].char_end]
    total = (await counter.count_many([raw]))[0]
    if total <= max_tokens:
        return [_block_from_nodes(source, nodes, total)]

    if heading_level <= 6:
        sections = _partition_at_heading(nodes, heading_level)
        if len(sections) > 1 or any(
            node.kind == "heading" and node.heading_level == heading_level for node in nodes
        ):
            output: list[MarkdownAssetBlock] = []
            for section in sections:
                output.extend(
                    await _split_by_heading(
                        source,
                        section,
                        counter,
                        max_tokens=max_tokens,
                        heading_level=heading_level + 1,
                    )
                )
            if output:
                return output
        return await _split_by_heading(
            source,
            nodes,
            counter,
            max_tokens=max_tokens,
            heading_level=heading_level + 1,
        )
    return await _split_content_nodes(source, nodes, counter, max_tokens=max_tokens)


def _partition_at_heading(nodes: list[MarkdownNode], level: int) -> list[list[MarkdownNode]]:
    sections: list[list[MarkdownNode]] = []
    current: list[MarkdownNode] = []
    for node in nodes:
        if node.kind == "heading" and node.heading_level == level and current:
            sections.append(current)
            current = []
        current.append(node)
    if current:
        sections.append(current)
    return sections


async def _split_content_nodes(
    source: str,
    nodes: list[MarkdownNode],
    counter: TokenCounter,
    *,
    max_tokens: int,
) -> list[MarkdownAssetBlock]:
    atoms: list[MarkdownNode] = []
    for node in nodes:
        if node.kind in {"heading", "horizontal_rule"}:
            continue
        if node.token_count <= max_tokens:
            atoms.append(node)
            continue
        if node.kind == "paragraph":
            atoms.extend(await _split_paragraph(node, counter, max_tokens=max_tokens))
        elif node.kind == "list":
            split_items = await _split_list(node, counter, max_tokens=max_tokens)
            atoms.extend(split_items or [node])
        else:
            atoms.append(node)

    groups: list[list[MarkdownNode]] = []
    current: list[MarkdownNode] = []
    current_tokens = 0
    for atom in atoms:
        if current and current_tokens + atom.token_count > max_tokens:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(atom)
        current_tokens += atom.token_count
        if atom.token_count > max_tokens:
            groups.append(current)
            current = []
            current_tokens = 0
    if current:
        groups.append(current)

    blocks: list[MarkdownAssetBlock] = []
    texts = [source[group[0].char_start : group[-1].char_end] for group in groups]
    exact_counts = await counter.count_many(texts)
    for group, raw, exact in zip(groups, texts, exact_counts, strict=True):
        oversized = exact > max_tokens and len(group) == 1
        reason = _oversized_reason(group[0]) if oversized else None
        blocks.append(
            MarkdownAssetBlock(
                char_start=group[0].char_start,
                char_end=group[-1].char_end,
                heading_path=_common_heading_path(group),
                raw=raw,
                token_count=exact,
                oversized=oversized,
                oversized_reason=reason,
            )
        )
    return blocks


async def _split_paragraph(
    node: MarkdownNode,
    counter: TokenCounter,
    *,
    max_tokens: int,
) -> list[MarkdownNode]:
    spans = _sentence_spans(node.raw)
    return await _nodes_from_spans(node, spans, counter, max_tokens=max_tokens)


async def _split_list(
    node: MarkdownNode,
    counter: TokenCounter,
    *,
    max_tokens: int,
) -> list[MarkdownNode]:
    pattern = r"(?m)^(?: {0,3})(?:[-+*]|\d+[.)])\s+"
    starts = [match.start() for match in re.finditer(pattern, node.raw)]
    if len(starts) <= 1:
        return []
    spans = list(zip(starts, starts[1:] + [len(node.raw)], strict=True))
    return await _nodes_from_spans(node, spans, counter, max_tokens=max_tokens)


async def _nodes_from_spans(
    parent: MarkdownNode,
    spans: list[tuple[int, int]],
    counter: TokenCounter,
    *,
    max_tokens: int,
) -> list[MarkdownNode]:
    texts = [parent.raw[start:end] for start, end in spans]
    counts = await counter.count_many(texts)
    output: list[MarkdownNode] = []
    for (start, end), text, count in zip(spans, texts, counts, strict=True):
        output.append(
            MarkdownNode(
                kind=parent.kind,
                char_start=parent.char_start + start,
                char_end=parent.char_start + end,
                raw=text,
                heading_path=parent.heading_path,
                token_count=count,
            )
        )
    return output


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    boundaries = [match.end() for match in re.finditer(r"[。！？!?；;]+(?:\s+|$)|[.!?]+\s+", text)]
    starts = [0, *boundaries]
    ends = [*boundaries, len(text)]
    return [(start, end) for start, end in zip(starts, ends, strict=True) if text[start:end]]


def _block_from_nodes(
    source: str,
    nodes: list[MarkdownNode],
    token_count: int,
) -> MarkdownAssetBlock:
    return MarkdownAssetBlock(
        char_start=nodes[0].char_start,
        char_end=nodes[-1].char_end,
        heading_path=_common_heading_path(nodes),
        raw=source[nodes[0].char_start : nodes[-1].char_end],
        token_count=token_count,
    )


def _common_heading_path(nodes: list[MarkdownNode]) -> list[str]:
    paths = [node.heading_path for node in nodes if node.kind not in {"heading", "horizontal_rule"}]
    if not paths:
        return []
    prefix = list(paths[0])
    for path in paths[1:]:
        while prefix and path[: len(prefix)] != prefix:
            prefix.pop()
    return prefix


def _has_content(nodes: list[MarkdownNode]) -> bool:
    return any(
        node.kind not in {"heading", "horizontal_rule"} and node.raw.strip()
        for node in nodes
    )


def _oversized_reason(node: MarkdownNode) -> str:
    reasons = {
        "code_fence": "indivisible_code_fence",
        "table": "indivisible_table",
        "list": "indivisible_list_item",
    }
    return reasons.get(node.kind, "indivisible_content_node")


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    for match in re.finditer("\n", source):
        offsets.append(match.end())
    if offsets[-1] != len(source):
        offsets.append(len(source))
    return offsets


async def _merge_short_blocks(
    source: str,
    blocks: list[MarkdownAssetBlock],
    counter: TokenCounter,
    *,
    max_tokens: int,
    min_tokens: int = 50,
) -> list[MarkdownAssetBlock]:
    merged: list[MarkdownAssetBlock] = []
    for block in blocks:
        if not merged:
            merged.append(block)
            continue
        previous = merged[-1]
        same_parent = previous.heading_path[:-1] == block.heading_path[:-1]
        eligible = previous.token_count < min_tokens or block.token_count < min_tokens
        if not same_parent or not eligible or previous.oversized or block.oversized:
            merged.append(block)
            continue
        raw = source[previous.char_start : block.char_end]
        combined_count = (await counter.count_many([raw]))[0]
        if combined_count > max_tokens:
            merged.append(block)
            continue
        merged[-1] = MarkdownAssetBlock(
            char_start=previous.char_start,
            char_end=block.char_end,
            heading_path=_shared_prefix(previous.heading_path, block.heading_path),
            raw=raw,
            token_count=combined_count,
        )
    return merged


def _shared_prefix(left: list[str], right: list[str]) -> list[str]:
    prefix: list[str] = []
    for left_item, right_item in zip(left, right, strict=False):
        if left_item != right_item:
            break
        prefix.append(left_item)
    return prefix


def _read_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")
