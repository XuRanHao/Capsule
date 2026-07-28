import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path


class VideoToolingError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class VideoSegment:
    segment_index: int
    start_ms: int
    end_ms: int
    output_path: Path
    preview_path: Path


class VideoParser:
    """FFmpeg/PySceneDetect adapter boundary for the segmentation milestone."""

    def __init__(self, concurrency: int) -> None:
        self._semaphore = asyncio.Semaphore(concurrency)

    @staticmethod
    def check_dependencies() -> list[str]:
        missing = []
        for executable in ("ffmpeg", "ffprobe"):
            if shutil.which(executable) is None:
                missing.append(executable)
        return missing

    async def segment(self, source: Path, output_dir: Path) -> list[VideoSegment]:
        async with self._semaphore:
            raise NotImplementedError(
                f"video segmentation adapter is not wired yet: {source} -> {output_dir}"
            )
