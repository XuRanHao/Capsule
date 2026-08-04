"""Extract document-embedded images and optionally run local OCR.

The module intentionally has no dependency on the import pipeline.  A document
parser can turn :class:`DocumentImage` values into image child ``AssetDraft``
objects, while retaining stable source anchors for the original document.
"""

from __future__ import annotations

import hashlib
import io
import mimetypes
import os
import posixpath
import tempfile
import threading
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import unquote

from PIL import Image, UnidentifiedImageError

MAX_DOCUMENT_IMAGE_BYTES = 64 * 1024 * 1024
MAX_DOCX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_DOCX_COMPRESSION_RATIO = 200
MAX_DOCX_MEMBER_COUNT = 10_000


@dataclass(frozen=True, slots=True)
class OcrText:
    """Reliable text recognised from one embedded image."""

    text: str
    confidence: float
    line_count: int


@dataclass(frozen=True, slots=True)
class DocumentImage:
    """A deduplicated, locally materialised image from a source document."""

    path: Path
    content_hash: str
    mime_type: str
    width: int
    height: int
    source_locator: dict[str, Any]
    file_info: dict[str, Any]
    ocr: OcrText | None = None

    @property
    def derived_file_uri(self) -> str:
        return self.path.resolve().as_uri()


@dataclass(frozen=True, slots=True)
class MediaExtraction:
    """Result of extracting images from one document."""

    images: list[DocumentImage] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)


class OcrEngine(Protocol):
    """A local OCR backend.  It may return lines or an engine-native result."""

    def __call__(self, image_path: Path) -> object: ...


class RapidOcrEngine:
    """Thin lazy adapter for ``rapidocr``'s local ONNX Runtime backend.

    Importing it only when OCR is requested keeps Markdown-only imports fast and
    makes the parser testable without optional OCR wheels installed.
    """

    def __init__(self) -> None:
        self._engine: object | None = None
        self._lock = threading.Lock()

    def __call__(self, image_path: Path) -> object:
        # Parsing is concurrent, while one local ONNX session is intentionally
        # shared. The lock also prevents duplicate lazy model initialisation.
        with self._lock:
            if self._engine is None:
                try:
                    from rapidocr import RapidOCR
                except ImportError as exc:  # pragma: no cover - optional dependency
                    raise RuntimeError(
                        "local OCR requires the optional 'rapidocr' dependency"
                    ) from exc
                self._engine = RapidOCR()
            engine = self._engine
            if not callable(engine):  # pragma: no cover - defensive broken dependency
                raise RuntimeError("RapidOCR did not initialise a callable engine")
            return engine(str(image_path))


class DocumentMediaExtractor:
    """Materialise useful DOCX/PDF images into a deterministic local cache.

    Exact-byte de-duplication happens per source document.  When the same image
    is referenced multiple times its ``source_locator`` retains every origin,
    so callers do not lose page/relationship provenance.
    """

    def __init__(
        self,
        output_root: Path,
        *,
        ocr_engine: OcrEngine | None = None,
        min_width: int = 32,
        min_height: int = 32,
        min_area: int = 4096,
        min_ocr_confidence: float = 0.65,
        min_ocr_characters: int = 2,
    ) -> None:
        if min_width < 1 or min_height < 1 or min_area < 1:
            raise ValueError("minimum image dimensions must be positive")
        if not 0 <= min_ocr_confidence <= 1:
            raise ValueError("min_ocr_confidence must be between zero and one")
        self._output_root = output_root
        self._ocr_engine = ocr_engine
        self._min_width = min_width
        self._min_height = min_height
        self._min_area = min_area
        self._min_ocr_confidence = min_ocr_confidence
        self._min_ocr_characters = min_ocr_characters

    def extract(self, document_path: Path) -> MediaExtraction:
        """Extract images for a DOCX or PDF path.

        Optional libraries are imported only inside the corresponding format
        reader.  Unsupported suffixes return an empty extraction rather than
        making a general document ingestion job fail.
        """

        suffix = document_path.suffix.lower()
        if suffix == ".docx":
            validate_docx_package(document_path)
            candidates = _extract_docx_candidates(document_path)
        elif suffix == ".pdf":
            candidates = _extract_pdf_candidates(document_path)
        else:
            return MediaExtraction()
        return self._materialize(document_path, candidates)

    def _materialize(
        self,
        document_path: Path,
        candidates: Iterable[_ImageCandidate],
    ) -> MediaExtraction:
        document_hash = _sha256_path(document_path)
        target_dir = self._output_root / document_hash
        images_by_hash: dict[str, DocumentImage] = {}
        skipped: list[dict[str, Any]] = []

        for candidate in candidates:
            if len(candidate.content) > MAX_DOCUMENT_IMAGE_BYTES:
                skipped.append({"source_locator": candidate.locator, "reason": "too_large"})
                continue
            extraction_error = candidate.locator.get("extraction_error")
            if extraction_error:
                skipped.append(
                    {"source_locator": candidate.locator, "reason": "extraction_failed"}
                )
                continue
            content_hash = hashlib.sha256(candidate.content).hexdigest()
            existing = images_by_hash.get(content_hash)
            if existing is not None:
                occurrences = existing.source_locator.setdefault("occurrences", [])
                if isinstance(occurrences, list):
                    occurrences.append(candidate.locator)
                continue

            inspected = _inspect_image(
                candidate.content,
                min_width=self._min_width,
                min_height=self._min_height,
                min_area=self._min_area,
            )
            if inspected is None:
                skipped.append({"source_locator": candidate.locator, "reason": "invalid_image"})
                continue
            if isinstance(inspected, str):
                skipped.append({"source_locator": candidate.locator, "reason": inspected})
                continue

            suffix, mime_type, width, height = inspected
            target_dir.mkdir(parents=True, exist_ok=True)
            resolved_root = self._output_root.expanduser().resolve()
            resolved_target = target_dir.resolve()
            if not resolved_target.is_relative_to(resolved_root):
                raise RuntimeError("document media target escaped the configured root")
            target_dir = resolved_target
            image_path = target_dir / f"{content_hash}{suffix}"
            if not image_path.exists():
                _atomic_write(image_path, candidate.content)

            locator = dict(candidate.locator)
            locator["occurrences"] = [dict(candidate.locator)]
            ocr, ocr_info = self._run_ocr(image_path)
            image = DocumentImage(
                path=image_path,
                content_hash=content_hash,
                mime_type=mime_type,
                width=width,
                height=height,
                source_locator=locator,
                file_info={
                    "width": width,
                    "height": height,
                    "mime_type": mime_type,
                    "file_size_bytes": len(candidate.content),
                    "embedded_in_document": True,
                    "ocr": ocr_info,
                },
                ocr=ocr,
            )
            images_by_hash[content_hash] = image

        return MediaExtraction(images=list(images_by_hash.values()), skipped=skipped)

    def _run_ocr(self, image_path: Path) -> tuple[OcrText | None, dict[str, Any]]:
        if self._ocr_engine is None:
            return None, {"status": "not_requested"}
        try:
            lines = _normalise_ocr_lines(self._ocr_engine(image_path))
        except Exception as exc:  # OCR must not make document extraction fail.
            return None, {"status": "failed", "error": str(exc)[:300]}
        usable = [(text, score) for text, score in lines if text.strip()]
        if not usable:
            return None, {"status": "empty", "line_count": 0}
        text = "\n".join(text.strip() for text, _ in usable)
        confidence = sum(score for _, score in usable) / len(usable)
        info = {
            "status": "accepted"
            if confidence >= self._min_ocr_confidence and len(text) >= self._min_ocr_characters
            else "filtered",
            "confidence": round(confidence, 6),
            "line_count": len(usable),
        }
        if info["status"] != "accepted":
            return None, info
        return OcrText(text=text, confidence=confidence, line_count=len(usable)), info


@dataclass(frozen=True, slots=True)
class _ImageCandidate:
    content: bytes
    locator: dict[str, Any]


def _extract_docx_candidates(document_path: Path) -> list[_ImageCandidate]:
    """Read OOXML media and retain part/relationship/order anchors.

    Headers, footers, comments, and footnotes have independent relationship
    parts, so inspecting only ``word/document.xml`` silently misses images
    which are visible in a Word document.
    """

    candidates: list[_ImageCandidate] = []
    with zipfile.ZipFile(document_path) as archive:
        names = set(archive.namelist())
        word_parts = sorted(
            name
            for name in names
            if name.startswith("word/") and name.endswith(".xml") and "/_rels/" not in name
        )
        order = 0
        for part in word_parts:
            rels_name = _docx_relationship_part(part)
            relationships: dict[str, str] = {}
            if rels_name in names:
                relationships = _docx_relationships(
                    archive.read(rels_name),
                    document_part=part,
                )
            embedded_ids = _docx_embedded_relationship_ids(archive.read(part))
            for relationship_id in embedded_ids:
                target = relationships.get(relationship_id)
                if target is None or target not in names:
                    continue
                candidates.append(
                    _ImageCandidate(
                        content=archive.read(target),
                        locator={
                            "type": "document_embedded_image",
                            "document_format": "docx",
                            "order": order,
                            "relationship_id": relationship_id,
                            "document_part": part,
                            "package_path": target,
                        },
                    )
                )
                order += 1
    return candidates


def validate_docx_package(document_path: Path) -> None:
    """Reject oversized or bomb-like OOXML packages before conversion."""

    with zipfile.ZipFile(document_path) as archive:
        members = archive.infolist()
        if len(members) > MAX_DOCX_MEMBER_COUNT:
            raise ValueError("DOCX package contains too many members")
        total_uncompressed = sum(member.file_size for member in members)
        if total_uncompressed > MAX_DOCX_UNCOMPRESSED_BYTES:
            raise ValueError("DOCX package expands beyond the safe size limit")
        for member in members:
            if member.file_size > MAX_DOCUMENT_IMAGE_BYTES and member.filename.startswith(
                "word/media/"
            ):
                raise ValueError("DOCX package contains an oversized media member")
            if (
                member.file_size >= 1024 * 1024
                and member.file_size > max(1, member.compress_size) * MAX_DOCX_COMPRESSION_RATIO
            ):
                raise ValueError("DOCX package has an unsafe compression ratio")


def _extract_pdf_candidates(document_path: Path) -> list[_ImageCandidate]:
    """Extract raster PDF XObjects with PyMuPDF, imported lazily."""

    try:
        import fitz  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "PDF image extraction requires the optional 'pymupdf' dependency"
        ) from exc

    candidates: list[_ImageCandidate] = []
    pdf = fitz.open(document_path)
    try:
        for page_index, page in enumerate(pdf):
            try:
                page_images = page.get_images(full=True)
            except Exception as exc:
                candidates.append(
                    _ImageCandidate(
                        content=b"",
                        locator={
                            "type": "document_embedded_image",
                            "document_format": "pdf",
                            "page": page_index + 1,
                            "extraction_error": str(exc)[:300],
                        },
                    )
                )
                page_images = []
            for order, image in enumerate(page_images):
                xref = int(image[0])
                locator = {
                    "type": "document_embedded_image",
                    "document_format": "pdf",
                    "page": page_index + 1,
                    "order": order,
                    "xref": xref,
                }
                get_image_rects = getattr(page, "get_image_rects", None)
                if callable(get_image_rects):
                    try:
                        locator["bboxes"] = [
                            [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]
                            for rect in get_image_rects(xref)
                        ]
                    except Exception:
                        pass
                try:
                    extracted = pdf.extract_image(xref)
                except Exception as exc:
                    candidates.append(
                        _ImageCandidate(
                            content=b"",
                            locator={**locator, "extraction_error": str(exc)[:300]},
                        )
                    )
                    continue
                content = extracted.get("image")
                if not isinstance(content, bytes):
                    continue
                candidates.append(
                    _ImageCandidate(
                        content=content,
                        locator={
                            **locator,
                            "extension": str(extracted.get("ext") or ""),
                        },
                    )
                )
            get_text = getattr(page, "get_text", None)
            if callable(get_text):
                try:
                    page_dict = get_text("dict")
                except Exception:
                    page_dict = None
                blocks = page_dict.get("blocks") if isinstance(page_dict, dict) else None
                if isinstance(blocks, list):
                    for block_index, block in enumerate(blocks):
                        if not isinstance(block, dict) or block.get("type") != 1:
                            continue
                        content = block.get("image")
                        if not isinstance(content, bytes):
                            continue
                        bbox = block.get("bbox")
                        locator = {
                            "type": "document_embedded_image",
                            "document_format": "pdf",
                            "page": page_index + 1,
                            "order": block_index,
                            "source": "page_image_block",
                            "extension": str(block.get("ext") or ""),
                        }
                        if isinstance(bbox, tuple | list) and len(bbox) == 4:
                            locator["bbox"] = [float(value) for value in bbox]
                        candidates.append(_ImageCandidate(content=content, locator=locator))
    finally:
        pdf.close()
    return candidates


def _docx_relationship_part(part: str) -> str:
    path = Path(part)
    return str(path.parent / "_rels" / f"{path.name}.rels")


def _docx_relationships(content: bytes, *, document_part: str) -> dict[str, str]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return {}
    relationships: dict[str, str] = {}
    for element in root.iter():
        if _xml_local_name(element.tag) != "Relationship":
            continue
        relationship_id = _xml_attribute(element, "Id")
        target = _xml_attribute(element, "Target")
        target_mode = _xml_attribute(element, "TargetMode")
        if relationship_id and target and (target_mode or "").lower() != "external":
            relationships[relationship_id] = _normalise_docx_target(document_part, target)
    return relationships


def _docx_embedded_relationship_ids(content: bytes) -> list[str]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []
    relationship_ids: list[str] = []
    for element in root.iter():
        local_name = _xml_local_name(element.tag)
        if local_name not in {"blip", "imagedata"}:
            continue
        relationship_id = _xml_attribute(
            element,
            "embed" if local_name == "blip" else "id",
        )
        if relationship_id:
            relationship_ids.append(relationship_id)
    return relationship_ids


def _xml_attribute(element: ET.Element, local_name: str) -> str | None:
    return next(
        (value for key, value in element.attrib.items() if _xml_local_name(key) == local_name),
        None,
    )


def _xml_local_name(name: str) -> str:
    return name.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _normalise_docx_target(part: str, target: str) -> str:
    decoded = unquote(target).replace("\\", "/")
    if decoded.startswith("/"):
        return posixpath.normpath(decoded).lstrip("/")
    return posixpath.normpath(posixpath.join(posixpath.dirname(part), decoded))


def _inspect_image(
    content: bytes,
    *,
    min_width: int,
    min_height: int,
    min_area: int,
) -> tuple[str, str, int, int] | str | None:
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            width, height = image.size
            if width < min_width or height < min_height or width * height < min_area:
                return "too_small"
            if _is_fully_transparent(image):
                return "fully_transparent"
            if _is_nearly_blank(image):
                return "nearly_blank"
            image_format = image.format or ""
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError):
        return None
    mime_type = Image.MIME.get(image_format) or mimetypes.guess_type(
        f"image.{image_format.lower()}"
    )[0]
    if mime_type is None:
        mime_type = "application/octet-stream"
    suffix = _image_suffix(image_format, mime_type)
    return suffix, mime_type, width, height


def _is_fully_transparent(image: Image.Image) -> bool:
    if "A" not in image.getbands():
        return False
    alpha = image.getchannel("A")
    extrema = alpha.getextrema()
    return bool(extrema and extrema[1] == 0)


def _is_nearly_blank(image: Image.Image) -> bool:
    """Filter whitespace/spacer images without rejecting solid-colour artwork."""

    sample = image.convert("L")
    sample.thumbnail((96, 96))
    extrema = cast(tuple[int, int] | None, sample.getextrema())
    return bool(extrema and extrema[0] >= 250)


def _image_suffix(image_format: str, mime_type: str) -> str:
    extension = Image.registered_extensions()
    for suffix, registered_format in extension.items():
        if registered_format == image_format:
            return suffix.lower()
    guessed = mimetypes.guess_extension(mime_type)
    return guessed or ".img"


def _normalise_ocr_lines(output: object) -> list[tuple[str, float]]:
    """Normalise RapidOCR-style and lightweight test-engine results."""

    txts = getattr(output, "txts", None)
    scores = getattr(output, "scores", None)
    if isinstance(txts, Sequence) and not isinstance(txts, str):
        score_values = (
            scores if isinstance(scores, Sequence) and not isinstance(scores, str) else []
        )
        return [
            (str(text), _as_confidence(score_values[index] if index < len(score_values) else 0.0))
            for index, text in enumerate(txts)
        ]
    if isinstance(output, Sequence) and not isinstance(output, str | bytes):
        normalised: list[tuple[str, float]] = []
        for item in output:
            line = _normalise_ocr_line(item)
            if line is not None:
                normalised.append(line)
        return normalised
    return []


def _normalise_ocr_line(item: object) -> tuple[str, float] | None:
    if isinstance(item, dict):
        text = item.get("text") or item.get("txt")
        score = item.get("score") or item.get("confidence")
        return (str(text), _as_confidence(score)) if text is not None else None
    if isinstance(item, Sequence) and not isinstance(item, str | bytes):
        if len(item) >= 3 and isinstance(item[1], str):
            return item[1], _as_confidence(item[2])
        if len(item) >= 2 and isinstance(item[0], str):
            return item[0], _as_confidence(item[1])
    return None


def _as_confidence(value: object) -> float:
    try:
        return min(1.0, max(0.0, float(str(value))))
    except (TypeError, ValueError):
        return 0.0


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    """Publish one content-addressed artifact without exposing partial bytes."""

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary_name = temporary.name
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
