"""Render compact context for the folder structure imported by a user."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol


class ImportedAssetPath(Protocol):
    """The only Asset field used to build an imported workspace tree."""

    @property
    def source_relative_path(self) -> str: ...


@dataclass(slots=True)
class _DirectoryNode:
    asset_count: int = 0
    directories: dict[str, _DirectoryNode] = field(default_factory=dict)
    files: Counter[str] = field(default_factory=Counter)


def render_imported_workspace_tree(
    assets: Iterable[ImportedAssetPath],
    *,
    max_files_per_directory: int = 3,
) -> str:
    """Return a stable, compact tree derived only from ``source_relative_path``.

    Directory counts include every Asset recursively. Files are grouped by their
    relative path because one imported source file can produce several Assets.
    Only a few direct file examples are rendered per directory to keep the result
    suitable for repeated use as model context.
    """
    if max_files_per_directory < 0:
        raise ValueError("max_files_per_directory must be at least zero")

    root = _DirectoryNode()
    for asset in assets:
        parts = _path_parts(asset.source_relative_path)
        root.asset_count += 1
        node = root
        for directory_name in parts[:-1]:
            node = node.directories.setdefault(directory_name, _DirectoryNode())
            node.asset_count += 1
        node.files[parts[-1]] += 1

    lines = [f"./ [{_asset_label(root.asset_count)}]"]
    _render_children(
        root,
        lines=lines,
        prefix="",
        max_files_per_directory=max_files_per_directory,
    )
    return "\n".join(lines)


def _path_parts(path: str) -> tuple[str, ...]:
    # Source paths are persisted as relative paths. Accept Windows separators so
    # imports created on another platform still preserve their directory levels.
    parts = tuple(part for part in path.replace("\\", "/").split("/") if part)
    return parts or ("(missing source_relative_path)",)


def _render_children(
    node: _DirectoryNode,
    *,
    lines: list[str],
    prefix: str,
    max_files_per_directory: int,
) -> None:
    directory_items = sorted(node.directories.items(), key=_named_item_sort_key)
    file_items = sorted(node.files.items(), key=_named_item_sort_key)
    visible_files = file_items[:max_files_per_directory]
    omitted_files = file_items[max_files_per_directory:]

    items: list[tuple[str, str, object]] = [
        ("directory", name, child) for name, child in directory_items
    ]
    items.extend(("file", name, count) for name, count in visible_files)
    if omitted_files:
        items.append(("omitted", "", omitted_files))

    for index, (kind, name, value) in enumerate(items):
        is_last = index == len(items) - 1
        connector = "└── " if is_last else "├── "
        if kind == "directory":
            child = value
            assert isinstance(child, _DirectoryNode)
            lines.append(f"{prefix}{connector}{name}/ [{_asset_label(child.asset_count)}]")
            _render_children(
                child,
                lines=lines,
                prefix=prefix + ("    " if is_last else "│   "),
                max_files_per_directory=max_files_per_directory,
            )
        elif kind == "file":
            count = value
            assert isinstance(count, int)
            suffix = f" [{_asset_label(count)}]" if count > 1 else ""
            lines.append(f"{prefix}{connector}{name}{suffix}")
        else:
            assert isinstance(value, list)
            omitted_asset_count = sum(count for _, count in value)
            lines.append(
                f"{prefix}{connector}… +{len(value)} files"
                f" [{_asset_label(omitted_asset_count)}]"
            )


def _named_item_sort_key(item: tuple[str, object]) -> tuple[str, str]:
    return item[0].casefold(), item[0]


def _asset_label(count: int) -> str:
    return f"{count} Asset" if count == 1 else f"{count} Assets"
