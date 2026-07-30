from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep

import numpy as np

from capsule.model_clients import mobileclip


def test_resident_worker_loads_mobileclip_once_across_calls(monkeypatch) -> None:
    loads = 0

    class FakeEmbedder:
        def __init__(self, *, model_path: Path | None, batch_size: int) -> None:
            nonlocal loads
            loads += 1
            assert model_path == Path("checkpoint.pt")
            assert batch_size == 12

        def embed(self, frames: list[np.ndarray]) -> np.ndarray:
            return np.ones((len(frames), 2), dtype=np.float32)

    monkeypatch.setattr(mobileclip, "MobileClipMpsEmbedder", FakeEmbedder)
    worker = mobileclip.ResidentMobileClipWorker(
        model_path=Path("checkpoint.pt"),
        batch_size=12,
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    assert worker.embed([frame]).shape == (1, 2)
    assert worker.embed([frame, frame]).shape == (2, 2)
    assert loads == 1


def test_resident_worker_prevents_concurrent_duplicate_loads(monkeypatch) -> None:
    loads = 0

    class SlowFakeEmbedder:
        def __init__(self, *, model_path: Path | None, batch_size: int) -> None:
            nonlocal loads
            loads += 1
            sleep(0.02)

        def embed(self, frames: list[np.ndarray]) -> np.ndarray:
            return np.ones((len(frames), 2), dtype=np.float32)

    monkeypatch.setattr(mobileclip, "MobileClipMpsEmbedder", SlowFakeEmbedder)
    worker = mobileclip.ResidentMobileClipWorker(model_path=None, batch_size=12)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: worker.embed([frame]), range(4)))

    assert all(result.shape == (1, 2) for result in results)
    assert loads == 1
