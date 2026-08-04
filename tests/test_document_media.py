from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from capsule.parsers.document_media import DocumentMediaExtractor, validate_docx_package


def test_docx_images_are_deduplicated_anchored_and_ocrd(tmp_path: Path) -> None:
    document = tmp_path / "example.docx"
    image = _png_bytes((30, 90, 160), size=(80, 60))
    spacer = _png_bytes((255, 255, 255), size=(80, 60))
    with zipfile.ZipFile(document, "w") as archive:
        archive.writestr(
            "word/document.xml",
            "<w:document xmlns:w='urn:word' xmlns:x='urn:drawing' "
            "xmlns:rel='urn:relationships'>"
            "<x:blip rel:embed='rId1'/><x:blip rel:embed='rId1'/>"
            "<x:blip rel:embed='rId2'/></w:document>",
        )
        archive.writestr(
            "word/_rels/document.xml.rels",
            "<Relationships><Relationship Target='media/image1.png' Id='rId1'/>"
            "<Relationship Target='media/spacer.png' Id='rId2'/></Relationships>",
        )
        archive.writestr("word/media/image1.png", image)
        archive.writestr("word/media/spacer.png", spacer)

    extractor = DocumentMediaExtractor(
        tmp_path / "derived",
        ocr_engine=lambda _: [[[], "可检索文字", 0.93]],
    )

    result = extractor.extract(document)

    assert len(result.images) == 1
    extracted = result.images[0]
    assert extracted.path.is_file()
    assert extracted.derived_file_uri.startswith("file://")
    assert extracted.ocr is not None
    assert extracted.ocr.text == "可检索文字"
    assert extracted.source_locator["relationship_id"] == "rId1"
    assert len(extracted.source_locator["occurrences"]) == 2
    assert result.skipped[0]["reason"] == "nearly_blank"


def test_ocr_low_confidence_is_retained_as_metadata_but_not_text(tmp_path: Path) -> None:
    document = _write_docx_with_one_image(tmp_path)
    result = DocumentMediaExtractor(
        tmp_path / "derived",
        ocr_engine=lambda _: [[[], "uncertain", 0.3]],
    ).extract(document)

    assert result.images[0].ocr is None
    assert result.images[0].file_info["ocr"] == {
        "status": "filtered",
        "confidence": 0.3,
        "line_count": 1,
    }


def test_ocr_failure_does_not_drop_the_image(tmp_path: Path) -> None:
    document = _write_docx_with_one_image(tmp_path)

    def broken_ocr(_: Path) -> object:
        raise RuntimeError("local inference failed")

    result = DocumentMediaExtractor(
        tmp_path / "derived",
        ocr_engine=broken_ocr,
    ).extract(document)

    assert len(result.images) == 1
    assert result.images[0].ocr is None
    assert result.images[0].file_info["ocr"] == {
        "status": "failed",
        "error": "local inference failed",
    }


def test_pdf_image_extraction_uses_page_and_xref(monkeypatch, tmp_path: Path) -> None:
    document = tmp_path / "slides.pdf"
    document.write_bytes(b"%PDF-test")
    image = _png_bytes((10, 20, 30), size=(64, 64))

    class FakePage:
        def get_images(self, *, full: bool):
            assert full is True
            return [(17, 0, 0, 0, 0, "", "", "")]

    class FakePdf:
        def __iter__(self):
            return iter([FakePage()])

        def extract_image(self, xref: int):
            assert xref == 17
            return {"image": image, "ext": "png"}

        def close(self) -> None:
            return None

    monkeypatch.setitem(sys.modules, "fitz", SimpleNamespace(open=lambda _: FakePdf()))

    result = DocumentMediaExtractor(tmp_path / "derived").extract(document)

    assert len(result.images) == 1
    locator = result.images[0].source_locator
    assert locator["document_format"] == "pdf"
    assert locator["page"] == 1
    assert locator["xref"] == 17


def test_pdf_bad_xobject_is_skipped_without_losing_other_images(
    monkeypatch,
    tmp_path: Path,
) -> None:
    document = tmp_path / "mixed.pdf"
    document.write_bytes(b"%PDF-test")
    image = _png_bytes((10, 20, 30), size=(64, 64))

    class FakePage:
        def get_images(self, *, full: bool):
            return [(17,), (18,)]

    class FakePdf:
        def __iter__(self):
            return iter([FakePage()])

        def extract_image(self, xref: int):
            if xref == 17:
                raise ValueError("dead XObject")
            return {"image": image, "ext": "png"}

        def close(self) -> None:
            return None

    monkeypatch.setitem(sys.modules, "fitz", SimpleNamespace(open=lambda _: FakePdf()))

    result = DocumentMediaExtractor(tmp_path / "derived").extract(document)

    assert len(result.images) == 1
    assert result.images[0].source_locator["xref"] == 18
    assert result.skipped[0]["reason"] == "extraction_failed"


def test_pdf_inline_image_block_is_extracted(monkeypatch, tmp_path: Path) -> None:
    document = tmp_path / "inline.pdf"
    document.write_bytes(b"%PDF-test")
    image = _png_bytes((50, 100, 150), size=(64, 64))

    class FakePage:
        def get_images(self, *, full: bool):
            return []

        def get_text(self, kind: str):
            assert kind == "dict"
            return {"blocks": [{"type": 1, "image": image, "bbox": (1, 2, 30, 40)}]}

    class FakePdf:
        def __iter__(self):
            return iter([FakePage()])

        def close(self) -> None:
            return None

    monkeypatch.setitem(sys.modules, "fitz", SimpleNamespace(open=lambda _: FakePdf()))

    result = DocumentMediaExtractor(tmp_path / "derived").extract(document)

    assert len(result.images) == 1
    assert result.images[0].source_locator["source"] == "page_image_block"
    assert result.images[0].source_locator["bbox"] == [1.0, 2.0, 30.0, 40.0]


def test_docx_vml_image_relationship_is_supported(tmp_path: Path) -> None:
    document = tmp_path / "legacy.docx"
    with zipfile.ZipFile(document, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="urn:w" xmlns:v="urn:v" xmlns:r="urn:r">'
            '<v:imagedata r:id="rId7"/></w:document>',
        )
        archive.writestr(
            "word/_rels/document.xml.rels",
            '<Relationships><Relationship Id="rId7" Target="media/legacy.png"/>'
            "</Relationships>",
        )
        archive.writestr("word/media/legacy.png", _png_bytes((20, 80, 160), size=(64, 64)))

    result = DocumentMediaExtractor(tmp_path / "derived").extract(document)

    assert len(result.images) == 1
    assert result.images[0].source_locator["relationship_id"] == "rId7"


def test_sparse_text_image_is_not_filtered_as_blank(tmp_path: Path) -> None:
    document = tmp_path / "sparse.docx"
    buffer = io.BytesIO()
    image = Image.new("RGB", (1200, 800), "white")
    ImageDraw.Draw(image).rectangle((100, 100, 1100, 116), fill="black")
    image.save(buffer, format="PNG")
    with zipfile.ZipFile(document, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="urn:w" xmlns:a="urn:a" xmlns:r="urn:r">'
            '<a:blip r:embed="rId1"/></w:document>',
        )
        archive.writestr(
            "word/_rels/document.xml.rels",
            '<Relationships><Relationship Id="rId1" Target="media/sparse.png"/>'
            "</Relationships>",
        )
        archive.writestr("word/media/sparse.png", buffer.getvalue())

    result = DocumentMediaExtractor(tmp_path / "derived").extract(document)

    assert len(result.images) == 1


def test_docx_preflight_rejects_unsafe_compression_ratio(tmp_path: Path) -> None:
    document = tmp_path / "bomb.docx"
    with zipfile.ZipFile(document, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", b"0" * (2 * 1024 * 1024))

    with pytest.raises(ValueError, match="compression ratio"):
        validate_docx_package(document)


def _write_docx_with_one_image(tmp_path: Path) -> Path:
    document = tmp_path / "example.docx"
    with zipfile.ZipFile(document, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="urn:w" xmlns:a="urn:a" xmlns:r="urn:r">'
            '<a:blip r:embed="rId1"/></w:document>',
        )
        archive.writestr(
            "word/_rels/document.xml.rels",
            '<Relationships><Relationship Id="rId1" Target="media/image1.png"/></Relationships>',
        )
        archive.writestr("word/media/image1.png", _png_bytes((40, 100, 200), size=(64, 64)))
    return document


def _png_bytes(color: tuple[int, int, int], *, size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size=size, color=color).save(buffer, format="PNG")
    return buffer.getvalue()
