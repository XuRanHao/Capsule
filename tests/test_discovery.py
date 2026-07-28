from pathlib import Path

from capsule.parsers.discovery import discover_files, sha256_file


def test_discover_files_filters_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "z.png").write_bytes(b"image")
    (tmp_path / "a.md").write_text("# Hello", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("ignore", encoding="utf-8")

    files = discover_files(tmp_path)

    assert [item.relative_path for item in files] == ["a.md", "z.png"]
    assert [item.extension for item in files] == [".md", ".png"]


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "asset.md"
    path.write_bytes(b"capsule")

    assert sha256_file(path) == sha256_file(path)
    assert len(sha256_file(path)) == 64
