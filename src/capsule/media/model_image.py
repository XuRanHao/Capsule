"""Prepare oversized source images for Ark model input without changing originals."""

import asyncio
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from math import sqrt

from PIL import Image, ImageOps

MODEL_IMAGE_TARGET_BYTES = 2 * 1024 * 1024
MODEL_IMAGE_MAX_EDGE = 1536


@dataclass(slots=True, frozen=True)
class PreparedModelImage:
    content: bytes
    mime_type: str
    width: int
    height: int
    resized: bool


class ModelImageCache:
    """Share bounded model-ready image derivatives across concurrent workloads."""

    def __init__(
        self,
        *,
        target_bytes: int = MODEL_IMAGE_TARGET_BYTES,
        max_edge: int = MODEL_IMAGE_MAX_EDGE,
        fixed_size: int | None = None,
        max_entries: int = 128,
    ) -> None:
        if (
            target_bytes <= 0
            or max_edge <= 0
            or max_entries <= 0
            or (fixed_size is not None and fixed_size <= 0)
        ):
            raise ValueError("model image cache limits must be positive")
        self._target_bytes = target_bytes
        self._max_edge = max_edge
        self._fixed_size = fixed_size
        self._max_entries = max_entries
        self._entries: OrderedDict[str, PreparedModelImage] = OrderedDict()
        self._in_flight: dict[str, asyncio.Task[PreparedModelImage]] = {}
        self._lock = asyncio.Lock()

    async def prepare(
        self,
        *,
        cache_key: str,
        mime_type: str,
        loader: Callable[[], bytes],
    ) -> PreparedModelImage:
        key = (
            f"{cache_key}:{mime_type.lower()}:{self._target_bytes}:"
            f"{self._max_edge}:{self._fixed_size}"
        )
        async with self._lock:
            cached = self._entries.pop(key, None)
            if cached is not None:
                self._entries[key] = cached
                return cached
            task = self._in_flight.get(key)
            if task is None:
                task = asyncio.create_task(
                    asyncio.to_thread(
                        _load_and_prepare,
                        loader,
                        mime_type,
                        self._target_bytes,
                        self._max_edge,
                        self._fixed_size,
                    )
                )
                self._in_flight[key] = task

        try:
            prepared = await asyncio.shield(task)
        except BaseException:
            async with self._lock:
                if task.done() and self._in_flight.get(key) is task:
                    self._in_flight.pop(key, None)
            raise

        async with self._lock:
            if self._in_flight.get(key) is task:
                self._in_flight.pop(key, None)
                self._entries[key] = prepared
                while len(self._entries) > self._max_entries:
                    self._entries.popitem(last=False)
        return prepared


def prepare_model_image(
    content: bytes,
    mime_type: str,
    *,
    target_bytes: int = MODEL_IMAGE_TARGET_BYTES,
    max_edge: int = MODEL_IMAGE_MAX_EDGE,
    fixed_size: int | None = None,
) -> PreparedModelImage:
    """Enforce both byte and dimension limits for a model-only image derivative."""
    if not content:
        raise ValueError("model image is empty")
    if target_bytes <= 0 or max_edge <= 0 or (fixed_size is not None and fixed_size <= 0):
        raise ValueError("model image limits must be positive")

    with Image.open(BytesIO(content)) as opened:
        image = ImageOps.exif_transpose(opened)
        width, height = image.size
        if fixed_size is not None:
            canvas = _fixed_canvas(image, fixed_size)
            encoded = _encode_to_target(canvas, target_bytes=target_bytes)
            return PreparedModelImage(
                content=encoded,
                mime_type="image/jpeg",
                width=fixed_size,
                height=fixed_size,
                resized=True,
            )
        if len(content) <= target_bytes and max(width, height) <= max_edge:
            return PreparedModelImage(
                content=content,
                mime_type=mime_type,
                width=width,
                height=height,
                resized=False,
            )

        byte_scale = (
            sqrt(target_bytes / len(content)) * 0.92 if len(content) > target_bytes else 1.0
        )
        scale = min(
            max_edge / max(width, height),
            byte_scale,
            1.0,
        )
        scale = max(scale, 1 / max(width, height))

        while True:
            resized_width = max(1, round(width * scale))
            resized_height = max(1, round(height * scale))
            resized = image.resize(
                (resized_width, resized_height),
                Image.Resampling.LANCZOS,
            )
            encoded, encoded_mime_type = _encode_image(resized)
            if (
                len(encoded) <= target_bytes and max(resized_width, resized_height) <= max_edge
            ) or (resized_width == 1 and resized_height == 1):
                return PreparedModelImage(
                    content=encoded,
                    mime_type=encoded_mime_type,
                    width=resized_width,
                    height=resized_height,
                    resized=True,
                )
            scale *= min(0.9, sqrt(target_bytes / len(encoded)) * 0.92)


def _load_and_prepare(
    loader: Callable[[], bytes],
    mime_type: str,
    target_bytes: int,
    max_edge: int,
    fixed_size: int | None,
) -> PreparedModelImage:
    return prepare_model_image(
        loader(),
        mime_type,
        target_bytes=target_bytes,
        max_edge=max_edge,
        fixed_size=fixed_size,
    )


def _encode_image(image: Image.Image, *, quality: int = 85) -> tuple[bytes, str]:
    output = BytesIO()
    _jpeg_compatible(image).save(
        output,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
    )
    return output.getvalue(), "image/jpeg"


def _fixed_canvas(image: Image.Image, size: int) -> Image.Image:
    """Fit without cropping, then center on a deterministic white square canvas."""
    contained = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    compatible = _jpeg_compatible(contained)
    offset = ((size - compatible.width) // 2, (size - compatible.height) // 2)
    canvas.paste(compatible, offset)
    return canvas


def _encode_to_target(image: Image.Image, *, target_bytes: int) -> bytes:
    encoded = b""
    for quality in (85, 75, 65, 55, 45, 35, 25):
        encoded, _ = _encode_image(image, quality=quality)
        if len(encoded) <= target_bytes:
            return encoded
    return encoded


def _jpeg_compatible(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")
