"""Deterministic, layout-aware chunking for unstructured UTF-8 text files."""

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

from capsule.enums import AssetType
from capsule.model_clients.tokenization import TokenCounter
from capsule.schemas import AssetDraft

_LIST_MARKER = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
_SENTENCE_BOUNDARY = re.compile(r"[。！？；;]+|[.!?]+(?:\s+|$)")


@dataclass(slots=True)
class TextNode:
    kind: str
    char_start: int
    char_end: int
    raw: str
    token_count: int = 0


@dataclass(slots=True)
class TextAssetBlock:
    char_start: int
    char_end: int
    raw: str
    token_count: int
    node_kinds: list[str]
    oversized: bool = False
    oversized_reason: str | None = None


#===========================================
#      TXT layout-aware splitting
#===========================================


class TextParser:
    """Convert unstructured text into stable text blocks without model inference.

    Empty lines separate natural paragraphs. Fenced/indented code, simple
    pipe/tabular tables, and list runs are recognized as indivisible units
    before length-based packing begins. Original character offsets are always
    retained; hard line wraps are never rewritten into the stored raw content.
    """

    async def assetize_file(
        self,
        path: Path,
        token_counter: TokenCounter,
        *,
        max_tokens: int = 1200,
    ) -> list[AssetDraft]:
        source = await asyncio.to_thread(path.read_text, encoding="utf-8")
        return await self.assetize(source, path.name, token_counter, max_tokens=max_tokens)

    async def assetize(
        self,
        source: str,
        file_name: str,
        token_counter: TokenCounter,
        *,
        max_tokens: int = 1200,
    ) -> list[AssetDraft]:
        nodes = _layout_nodes(source)
        if not nodes:
            return []
        counts = await token_counter.count_many([node.raw for node in nodes])
        for node, count in zip(nodes, counts, strict=True):
            node.token_count = count

        atoms: list[TextNode] = []
        for node in nodes:
            if node.token_count <= max_tokens:
                atoms.append(node)
            elif node.kind == "paragraph":
                atoms.extend(await _split_paragraph(node, token_counter, max_tokens=max_tokens))
            elif node.kind == "list":
                items = await _split_list(node, token_counter, max_tokens=max_tokens)
                atoms.extend(items or [node])
            else:
                atoms.append(node)

        blocks = await _pack_atoms(source, atoms, token_counter, max_tokens=max_tokens)
        return [
            AssetDraft(
                asset_type=AssetType.TEXT_BLOCK,
                file_name=file_name,
                source_locator={
                    "type": "text_range",
                    "block_index": index,
                    "char_start": block.char_start,
                    "char_end": block.char_end,
                },
                raw_content=block.raw,
                file_info={
                    "token_count": block.token_count,
                    "node_kinds": block.node_kinds,
                    "oversized": block.oversized,
                    "oversized_reason": block.oversized_reason,
                },
            )
            for index, block in enumerate(blocks)
        ]


def _layout_nodes(source: str) -> list[TextNode]:
    lines = _lines(source)
    nodes: list[TextNode] = []
    index = 0
    while index < len(lines):
        if _is_blank(lines[index].raw):
            index += 1
            continue
        if _is_fence_start(lines[index].raw):
            end_index = _fence_end(lines, index)
            nodes.append(_node_from_lines(source, lines, index, end_index, "code_fence"))
            index = end_index + 1
            continue
        if _is_indented_code(lines[index].raw):
            end_index = _indented_code_end(lines, index)
            nodes.append(_node_from_lines(source, lines, index, end_index, "code_fence"))
            index = end_index + 1
            continue

        end_index = index
        while end_index + 1 < len(lines):
            next_line = lines[end_index + 1]
            if _is_blank(next_line.raw) or _is_fence_start(next_line.raw):
                break
            if _is_indented_code(next_line.raw):
                break
            end_index += 1
        raw_lines = [line.raw for line in lines[index : end_index + 1]]
        nodes.append(
            _node_from_lines(
                source,
                lines,
                index,
                end_index,
                _group_kind(raw_lines),
            )
        )
        index = end_index + 1
    return nodes


@dataclass(slots=True, frozen=True)
class _TextLine:
    char_start: int
    char_end: int
    raw: str


def _lines(source: str) -> list[_TextLine]:
    output: list[_TextLine] = []
    offset = 0
    for raw in source.splitlines(keepends=True):
        output.append(_TextLine(char_start=offset, char_end=offset + len(raw), raw=raw))
        offset += len(raw)
    if source and not output:
        output.append(_TextLine(char_start=0, char_end=len(source), raw=source))
    return output


def _node_from_lines(
    source: str,
    lines: list[_TextLine],
    start_index: int,
    end_index: int,
    kind: str,
) -> TextNode:
    char_start = lines[start_index].char_start
    char_end = lines[end_index].char_end
    return TextNode(
        kind=kind,
        char_start=char_start,
        char_end=char_end,
        raw=source[char_start:char_end],
    )


def _is_blank(raw: str) -> bool:
    return not raw.strip()


def _is_fence_start(raw: str) -> bool:
    return raw.lstrip().startswith(("```", "~~~"))


def _fence_end(lines: list[_TextLine], start_index: int) -> int:
    marker = lines[start_index].raw.lstrip()[:3]
    for index in range(start_index + 1, len(lines)):
        if lines[index].raw.lstrip().startswith(marker):
            return index
    return len(lines) - 1


def _is_indented_code(raw: str) -> bool:
    return raw.startswith(("    ", "\t"))


def _indented_code_end(lines: list[_TextLine], start_index: int) -> int:
    index = start_index
    while index + 1 < len(lines):
        next_line = lines[index + 1]
        if not (_is_blank(next_line.raw) or _is_indented_code(next_line.raw)):
            break
        index += 1
    return index


def _group_kind(raw_lines: list[str]) -> str:
    visible = [line for line in raw_lines if not _is_blank(line)]
    if len(visible) >= 2 and all("|" in line or "\t" in line for line in visible):
        return "table"
    if len(visible) >= 1 and all(_LIST_MARKER.match(line) for line in visible):
        return "list"
    return "paragraph"


async def _split_paragraph(
    node: TextNode,
    counter: TokenCounter,
    *,
    max_tokens: int,
) -> list[TextNode]:
    spans = _sentence_spans(node.raw)
    return await _nodes_from_spans(node, spans, counter, max_tokens=max_tokens)


async def _split_list(
    node: TextNode,
    counter: TokenCounter,
    *,
    max_tokens: int,
) -> list[TextNode]:
    starts = [match.start() for match in re.finditer(r"(?m)^\s*(?:[-+*]|\d+[.)])\s+", node.raw)]
    if len(starts) <= 1:
        return []
    spans = list(zip(starts, starts[1:] + [len(node.raw)], strict=True))
    return await _nodes_from_spans(node, spans, counter, max_tokens=max_tokens)


async def _nodes_from_spans(
    parent: TextNode,
    spans: list[tuple[int, int]],
    counter: TokenCounter,
    *,
    max_tokens: int,
) -> list[TextNode]:
    texts = [parent.raw[start:end] for start, end in spans]
    counts = await counter.count_many(texts)
    return [
        TextNode(
            kind=parent.kind,
            char_start=parent.char_start + start,
            char_end=parent.char_start + end,
            raw=text,
            token_count=count,
        )
        for (start, end), text, count in zip(spans, texts, counts, strict=True)
    ]


async def _pack_atoms(
    source: str,
    atoms: list[TextNode],
    counter: TokenCounter,
    *,
    max_tokens: int,
) -> list[TextAssetBlock]:
    groups: list[list[TextNode]] = []
    current: list[TextNode] = []
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

    raw_blocks = [source[group[0].char_start : group[-1].char_end] for group in groups]
    exact_counts = await counter.count_many(raw_blocks)
    blocks: list[TextAssetBlock] = []
    for group, raw, token_count in zip(groups, raw_blocks, exact_counts, strict=True):
        oversized = token_count > max_tokens and len(group) == 1
        blocks.append(
            TextAssetBlock(
                char_start=group[0].char_start,
                char_end=group[-1].char_end,
                raw=raw,
                token_count=token_count,
                node_kinds=sorted({node.kind for node in group}),
                oversized=oversized,
                oversized_reason=_oversized_reason(group[0]) if oversized else None,
            )
        )
    return blocks


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    boundaries = [match.end() for match in _SENTENCE_BOUNDARY.finditer(text)]
    starts = [0, *boundaries]
    ends = [*boundaries, len(text)]
    return [
        (start, end)
        for start, end in zip(starts, ends, strict=True)
        if text[start:end].strip()
    ]


def _oversized_reason(node: TextNode) -> str:
    reasons = {
        "code_fence": "indivisible_code_fence",
        "table": "indivisible_table",
        "list": "indivisible_list_item",
    }
    return reasons.get(node.kind, "indivisible_sentence")
