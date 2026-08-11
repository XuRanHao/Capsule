"""Authorization checks for local video source references."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def validate_video_source_root(
    path: Path,
    *,
    import_root: Path,
    video_source_roots: Sequence[Path],
) -> None:
    """Reject a local video path that cannot be read through configured roots."""
    source = path.expanduser().resolve()
    for root in (import_root, *video_source_roots):
        try:
            source.relative_to(root.expanduser().resolve())
        except ValueError:
            continue
        return
    raise ValueError(
        "video source must be under import_root or CAPSULE_VIDEO_SOURCE_ROOTS"
    )
