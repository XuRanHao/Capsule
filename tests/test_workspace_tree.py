from dataclasses import dataclass

import pytest

from capsule.pipeline.workspace_tree import render_imported_workspace_tree


@dataclass(frozen=True)
class _Asset:
    source_relative_path: str


def test_renders_stable_directory_first_tree_with_recursive_asset_counts() -> None:
    assets = [
        _Asset("NPC/B人物/立绘.png"),
        _Asset("README.md"),
        _Asset("NPC/A人物/武器/长剑.png"),
        _Asset("NPC/A人物/服装.png"),
        _Asset("NPC/A人物/武器/长剑.png"),
    ]

    rendered = render_imported_workspace_tree(assets)

    assert rendered == """./ [5 Assets]
├── NPC/ [4 Assets]
│   ├── A人物/ [3 Assets]
│   │   ├── 武器/ [2 Assets]
│   │   │   └── 长剑.png [2 Assets]
│   │   └── 服装.png
│   └── B人物/ [1 Asset]
│       └── 立绘.png
└── README.md"""


def test_caps_direct_file_examples_and_summarizes_the_rest() -> None:
    assets = [
        _Asset("场景/c.png"),
        _Asset("场景/a.png"),
        _Asset("场景/e.png"),
        _Asset("场景/b.png"),
        _Asset("场景/d.png"),
        _Asset("场景/e.png"),
    ]

    rendered = render_imported_workspace_tree(assets, max_files_per_directory=2)

    assert rendered == """./ [6 Assets]
└── 场景/ [6 Assets]
    ├── a.png
    ├── b.png
    └── … +3 files [4 Assets]"""


def test_accepts_windows_separators_and_keeps_assets_without_a_path() -> None:
    rendered = render_imported_workspace_tree(
        [_Asset(r"角色\NPC\A.png"), _Asset("")]
    )

    assert rendered == """./ [2 Assets]
├── 角色/ [1 Asset]
│   └── NPC/ [1 Asset]
│       └── A.png
└── (missing source_relative_path)"""


def test_empty_input_and_zero_file_examples_are_supported() -> None:
    assert render_imported_workspace_tree([]) == "./ [0 Assets]"
    assert render_imported_workspace_tree(
        [_Asset("NPC/A.png")], max_files_per_directory=0
    ) == """./ [1 Asset]
└── NPC/ [1 Asset]
    └── … +1 files [1 Asset]"""


def test_rejects_negative_file_example_limit() -> None:
    with pytest.raises(ValueError, match="at least zero"):
        render_imported_workspace_tree([], max_files_per_directory=-1)
