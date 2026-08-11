import asyncio
from pathlib import Path

import pytest

from capsule.media.video_frames import (
    FFmpegVideoFrameExtractor,
    VideoFrameRequest,
    logical_video_frame_request,
)


def test_logical_video_frame_request_uses_bounded_unique_timestamps() -> None:
    request = logical_video_frame_request(
        source_uri="file:///library/original.mp4",
        file_info={
            "representative_frames": [
                {"timestamp_ms": 9_000},
                {"timestamp_ms": 12_000},
                {"timestamp_ms": 12_000},
                {"timestamp_ms": 15_000},
                {"timestamp_ms": 18_000},
                {"timestamp_ms": 21_000},
            ]
        },
        source_locator={"start_ms": 10_000, "end_ms": 20_000},
    )

    assert request.source_uri == "file:///library/original.mp4"
    assert request.timestamps_ms == (12_000, 15_000, 18_000)


def test_logical_video_frame_request_falls_back_to_segment_midpoint() -> None:
    request = logical_video_frame_request(
        source_uri="file:///library/original.mov",
        file_info={},
        source_locator={"start_ms": 2_000, "end_ms": 6_001},
    )

    assert request.timestamps_ms == (4_000,)


@pytest.mark.asyncio
async def test_ffmpeg_frame_extractor_returns_jpeg_without_output_file(tmp_path: Path) -> None:
    fixture = await asyncio.to_thread(Path("data/dev-fixtures/nature/hiking-trip.mp4").resolve)
    if not await asyncio.to_thread(fixture.is_file):
        pytest.skip("video fixture is unavailable")
    before = await asyncio.to_thread(lambda: set(tmp_path.iterdir()))

    frames = await FFmpegVideoFrameExtractor(concurrency=1).extract(
        VideoFrameRequest(source_uri=fixture.as_uri(), timestamps_ms=(1_000,))
    )

    assert len(frames) == 1
    assert frames[0].startswith(b"\xff\xd8")
    assert frames[0].endswith(b"\xff\xd9")
    assert await asyncio.to_thread(lambda: set(tmp_path.iterdir())) == before


@pytest.mark.asyncio
async def test_frame_extractor_coalesces_only_concurrent_identical_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extractor = FFmpegVideoFrameExtractor(concurrency=1)
    request = VideoFrameRequest(
        source_uri="file:///library/original.mp4",
        timestamps_ms=(1_000,),
    )
    calls = 0
    release = asyncio.Event()

    async def fake_extract(_request: VideoFrameRequest) -> list[bytes]:
        nonlocal calls
        calls += 1
        await release.wait()
        return [b"\xff\xd8shared"]

    monkeypatch.setattr(extractor, "_extract_uncached", fake_extract)
    first = asyncio.create_task(extractor.extract(request))
    second = asyncio.create_task(extractor.extract(request))
    await asyncio.sleep(0)
    release.set()

    assert await first == [b"\xff\xd8shared"]
    assert await second == [b"\xff\xd8shared"]
    assert calls == 1

    release.clear()
    third = asyncio.create_task(extractor.extract(request))
    await asyncio.sleep(0)
    release.set()
    assert await third == [b"\xff\xd8shared"]
    assert calls == 2
