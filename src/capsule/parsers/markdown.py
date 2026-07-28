from dataclasses import dataclass, field
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token

from capsule.schemas import SourceContext


@dataclass(slots=True)
class MarkdownTextBlock:
    block_index: int
    text: str
    heading_path: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MarkdownImageReference:
    image_path: str
    contexts: list[SourceContext] = field(default_factory=list)


@dataclass(slots=True)
class MarkdownParseResult:
    text_blocks: list[MarkdownTextBlock] = field(default_factory=list)
    image_references: list[MarkdownImageReference] = field(default_factory=list)


class MarkdownParser:
    """Extract text blocks and source context for referenced images."""

    def __init__(self) -> None:
        self._parser = MarkdownIt("commonmark")

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
                        )
                    )
            index += 1

        return result


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
