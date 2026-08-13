"""Resident EfficientAT acoustic encoder for the shared macOS MPS Worker."""

from __future__ import annotations

import contextlib
import importlib
import io
import os
import sys
import threading
from pathlib import Path
from typing import cast

import numpy as np

from capsule.parsers.video import VideoToolingError


class EfficientAtMpsEmbedder:
    """Load the official ``mn10_as`` checkpoint and expose only its audio tower."""

    def __init__(self, *, source_path: Path, model_name: str, batch_size: int) -> None:
        try:
            self._torch = importlib.import_module("torch")
            self._librosa = importlib.import_module("librosa")
        except ModuleNotFoundError as exc:
            raise VideoToolingError(
                "EfficientAT requires torch and librosa in the macOS media Worker environment"
            ) from exc
        if not self._torch.backends.mps.is_available():
            raise VideoToolingError("MPS is unavailable for EfficientAT")
        root = source_path.expanduser().resolve()
        if not (root / "models/mn/model.py").is_file():
            raise VideoToolingError(f"EfficientAT source tree does not exist: {root}")
        sys.path.insert(0, str(root))
        previous = Path.cwd()
        try:
            os.chdir(root)
            helpers = importlib.import_module("helpers.utils")
            model_module = importlib.import_module("models.mn.model")
            with contextlib.redirect_stdout(io.StringIO()):
                self._model = model_module.get_model(
                    width_mult=helpers.NAME_TO_WIDTH(model_name),
                    pretrained_name=model_name,
                    head_type="mlp",
                )
        finally:
            os.chdir(previous)
        self._model = self._model.eval().to("mps")
        self._batch_size = batch_size

    def embed(self, waveforms: list[np.ndarray], *, sample_rate: int) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(waveforms), self._batch_size):
            batch = waveforms[start : start + self._batch_size]
            specs = [self._log_mel(item, sample_rate=sample_rate) for item in batch]
            tensor = self._torch.from_numpy(np.stack(specs)).unsqueeze(1).to("mps")
            with self._torch.inference_mode():
                _, features = self._model(tensor)
                features = self._torch.nn.functional.normalize(features.float(), dim=-1)
            vectors.append(features.cpu().numpy().astype(np.float32))
        return np.concatenate(vectors, axis=0)

    def _log_mel(self, waveform: np.ndarray, *, sample_rate: int) -> np.ndarray:
        emphasized = waveform[1:] - 0.97 * waveform[:-1]
        power = self._librosa.feature.melspectrogram(
            y=emphasized,
            sr=sample_rate,
            n_fft=1024,
            win_length=800,
            hop_length=320,
            window="hann",
            center=True,
            power=2.0,
            n_mels=128,
            fmin=0.0,
            fmax=15_000.0,
            norm=None,
            htk=True,
        )
        return cast(
            np.ndarray,
            ((np.log(power + 1e-5) + 4.5) / 5.0).astype(np.float32),
        )


class ResidentEfficientAtWorker:
    """Lazily retain one EfficientAT model beside the resident MobileCLIP model."""

    def __init__(self, *, source_path: Path, model_name: str, batch_size: int) -> None:
        self._source_path = source_path
        self._model_name = model_name
        self._batch_size = batch_size
        self._embedder: EfficientAtMpsEmbedder | None = None
        self._lock = threading.Lock()

    def embed(self, waveforms: list[np.ndarray], *, sample_rate: int) -> np.ndarray:
        with self._lock:
            if self._embedder is None:
                self._embedder = EfficientAtMpsEmbedder(
                    source_path=self._source_path,
                    model_name=self._model_name,
                    batch_size=self._batch_size,
                )
            return self._embedder.embed(waveforms, sample_rate=sample_rate)
