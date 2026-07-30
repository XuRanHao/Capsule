from io import BytesIO
from os import urandom

from PIL import Image

from capsule.media.model_image import prepare_model_image


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
    assert prepared.mime_type == "image/png"
    assert prepared.resized is True
