"""Optional Apple MobileCLIP image encoder for the macOS MPS video Worker."""

import importlib
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from capsule.parsers.video import VideoToolingError


class MobileClipMpsEmbedder:
    """Lazily load MobileCLIP-S0 in a host Python environment with MPS support."""

    def __init__(self, *, model_path: Path | None, batch_size: int) -> None:
        try:
            torch = importlib.import_module("torch")
            mobileclip = importlib.import_module("mobileclip")
        except ModuleNotFoundError as exc:
            raise VideoToolingError(
                "MobileCLIP is not installed. Run this video import from the macOS MPS Worker "
                "environment with PyTorch and Apple's ml-mobileclip package installed."
            ) from exc
        if not torch.backends.mps.is_available():
            raise VideoToolingError(
                "MPS is unavailable in this Python runtime. Use a native Apple-silicon PyTorch "
                "environment with MPS enabled; the Docker app container cannot provide MPS."
            )

        checkpoint = (model_path or Path("data/models/mobileclip-s0/mobileclip_s0.pt")).expanduser()
        if not checkpoint.is_file():
            raise VideoToolingError(f"MobileCLIP-S0 checkpoint does not exist: {checkpoint}")

        self._torch: Any = torch
        self._batch_size = batch_size
        self._model, _, self._transform = mobileclip.create_model_and_transforms(
            "mobileclip_s0",
            pretrained=str(checkpoint),
            device="mps",
        )
        self._model.eval()

    def embed(self, frames: list[np.ndarray]) -> np.ndarray:
        if not frames:
            return np.empty((0, 512), dtype=np.float32)
        vectors: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(frames), self._batch_size):
                batch = [
                    self._transform(Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))).to(
                        "mps"
                    )
                    for frame in frames[start : start + self._batch_size]
                ]
                tensor = self._torch.stack(batch)
                vector = self._model.encode_image(tensor, normalize=True)
                vectors.append(vector.float().cpu().numpy())
        return np.concatenate(vectors, axis=0).astype(np.float32, copy=False)


class ResidentMobileClipWorker:
    """Keep one lazily loaded MobileCLIP model resident for the process lifetime.

    Video files are analyzed on worker threads.  The lock prevents concurrent
    first-use model construction and serializes access to the shared MPS model.
    """

    def __init__(self, *, model_path: Path | None, batch_size: int) -> None:
        self._model_path = model_path
        self._batch_size = batch_size
        self._embedder: MobileClipMpsEmbedder | None = None
        self._lock = threading.Lock()

    def embed(self, frames: list[np.ndarray]) -> np.ndarray:
        with self._lock:
            if self._embedder is None:
                self._embedder = MobileClipMpsEmbedder(
                    model_path=self._model_path,
                    batch_size=self._batch_size,
                )
            return self._embedder.embed(frames)
