import hashlib
from collections.abc import Iterable
from pathlib import Path

from capsule.schemas import DiscoveredFile

SUPPORTED_EXTENSIONS = frozenset({".md", ".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov"})


def discover_files(root: Path) -> list[DiscoveredFile]:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    paths: Iterable[Path]
    if root.is_file():
        paths = [root]
        relative_root = root.parent
    else:
        paths = (path for path in root.rglob("*") if path.is_file())
        relative_root = root

    discovered = [
        DiscoveredFile(
            path=str(path),
            relative_path=path.relative_to(relative_root).as_posix(),
            extension=path.suffix.lower(),
            size_bytes=path.stat().st_size,
        )
        for path in paths
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(discovered, key=lambda item: item.relative_path)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
