"""Prepare oversized source images for Ark model input without changing originals."""

from dataclasses import dataclass
from io import BytesIO
from math import sqrt

from PIL import Image, ImageOps

MODEL_IMAGE_TARGET_BYTES = 9 * 1024 * 1024
MODEL_IMAGE_MAX_EDGE = 3072


@dataclass(slots=True, frozen=True)
class PreparedModelImage:
    content: bytes
    mime_type: str
    width: int
    height: int
    resized: bool


def prepare_model_image(
    content: bytes,
    mime_type: str,
    *,
    target_bytes: int = MODEL_IMAGE_TARGET_BYTES,
    max_edge: int = MODEL_IMAGE_MAX_EDGE,
) -> PreparedModelImage:
    """Downscale oversized images until their encoded form is below the target."""
    if not content:
        raise ValueError("model image is empty")
    if target_bytes <= 0 or max_edge <= 0:
        raise ValueError("model image limits must be positive")

    with Image.open(BytesIO(content)) as opened:
        width, height = opened.size
        if len(content) <= target_bytes:
            return PreparedModelImage(
                content=content,
                mime_type=mime_type,
                width=width,
                height=height,
                resized=False,
            )

        image = ImageOps.exif_transpose(opened)
        width, height = image.size
        source_format = (opened.format or _format_for_mime_type(mime_type)).upper()
        scale = min(
            max_edge / max(width, height),
            sqrt(target_bytes / len(content)) * 0.92,
            0.95,
        )
        scale = max(scale, 1 / max(width, height))

        while True:
            resized_width = max(1, round(width * scale))
            resized_height = max(1, round(height * scale))
            resized = image.resize(
                (resized_width, resized_height),
                Image.Resampling.LANCZOS,
            )
            encoded, encoded_mime_type = _encode_image(resized, source_format)
            if len(encoded) <= target_bytes or (resized_width == 1 and resized_height == 1):
                return PreparedModelImage(
                    content=encoded,
                    mime_type=encoded_mime_type,
                    width=resized_width,
                    height=resized_height,
                    resized=True,
                )
            scale *= min(0.9, sqrt(target_bytes / len(encoded)) * 0.92)


def _encode_image(image: Image.Image, source_format: str) -> tuple[bytes, str]:
    output = BytesIO()
    if source_format in {"JPEG", "JPG"}:
        _jpeg_compatible(image).save(output, format="JPEG", quality=90, optimize=True)
        return output.getvalue(), "image/jpeg"
    if source_format == "WEBP":
        image.save(output, format="WEBP", quality=90, method=6)
        return output.getvalue(), "image/webp"
    image.save(output, format="PNG", optimize=True)
    return output.getvalue(), "image/png"


def _jpeg_compatible(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def _format_for_mime_type(mime_type: str) -> str:
    return {
        "image/jpeg": "JPEG",
        "image/jpg": "JPEG",
        "image/webp": "WEBP",
    }.get(mime_type.lower(), "PNG")
