import asyncio
from io import BytesIO
from os import urandom

import pytest
from PIL import Image

from capsule.media.model_image import ModelImageCache, prepare_model_image


def test_small_model_image_is_kept_unchanged() -> None:
    output = BytesIO()
    Image.new("RGB", (32, 24), "orange").save(output, format="PNG")
    original = output.getvalue()

    prepared = prepare_model_image(original, "image/png", target_bytes=1024)

    assert prepared.content == original
    assert prepared.mime_type == "image/png"
    assert prepared.width == 32
    assert prepared.height == 24
    assert prepared.resized is False


def test_oversized_model_image_is_downscaled_below_target() -> None:
    output = BytesIO()
    Image.frombytes("RGB", (512, 512), urandom(512 * 512 * 3)).save(
        output,
        format="PNG",
    )
    original = output.getvalue()

    prepared = prepare_model_image(
        original,
        "image/png",
        target_bytes=150 * 1024,
        max_edge=256,
    )

    assert len(prepared.content) <= 150 * 1024
    assert max(prepared.width, prepared.height) <= 256
    assert prepared.width < 512
    assert prepared.height < 512
    assert prepared.mime_type == "image/jpeg"
    assert prepared.resized is True


def test_model_image_enforces_edge_limit_even_when_source_is_below_byte_limit() -> None:
    output = BytesIO()
    Image.new("RGB", (4096, 256), "white").save(output, format="PNG")
    original = output.getvalue()
    assert len(original) < 2 * 1024 * 1024

    prepared = prepare_model_image(
        original,
        "image/png",
        target_bytes=2 * 1024 * 1024,
        max_edge=1536,
    )

    assert max(prepared.width, prepared.height) == 1536
    assert prepared.mime_type == "image/jpeg"
    assert prepared.resized is True


@pytest.mark.asyncio
async def test_model_image_cache_coalesces_concurrent_preparation() -> None:
    output = BytesIO()
    Image.new("RGB", (1024, 768), "navy").save(output, format="PNG")
    source = output.getvalue()
    loads = 0

    def load() -> bytes:
        nonlocal loads
        loads += 1
        return source

    cache = ModelImageCache(target_bytes=128 * 1024, max_edge=512, max_entries=2)
    first, second = await asyncio.gather(
        cache.prepare(cache_key="same-image", mime_type="image/png", loader=load),
        cache.prepare(cache_key="same-image", mime_type="image/png", loader=load),
    )

    assert loads == 1
    assert first is second
