from PIL import Image

from lao_document_ocr.models import BoundingBox
from lao_document_ocr.ocr.base import OcrEngine, RecognizedLine
from lao_document_ocr.pipeline import process_document


class FakeEngine(OcrEngine):
    def is_available(self) -> bool:
        return True

    def recognize(self, image: Image.Image) -> list[RecognizedLine]:
        return [
            RecognizedLine(
                text="Hello OCR",
                bbox=BoundingBox(x=10, y=10, width=120, height=24),
                confidence=0.99,
                block_id=1,
                paragraph_id=1,
                line_id=1,
            )
        ]


def test_process_image_with_fake_engine(tmp_path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (320, 180), "white").save(image_path)

    document = process_document(image_path, engine=FakeEngine())

    assert len(document.pages) == 1
    assert document.pages[0].blocks[0].text == "Hello OCR"
    assert document.metadata["engine"]["name"] == "FakeEngine"


def test_processing_can_be_cancelled_before_start(tmp_path) -> None:
    import pytest

    from lao_document_ocr.pipeline import DocumentProcessingCancelled

    image_path = tmp_path / "cancel.png"
    Image.new("RGB", (320, 180), "white").save(image_path)

    with pytest.raises(DocumentProcessingCancelled, match="cancelled"):
        process_document(
            image_path,
            engine=FakeEngine(),
            should_cancel=lambda: True,
        )


def test_processing_cancels_between_pdf_pages(tmp_path) -> None:
    import pymupdf
    import pytest

    from lao_document_ocr.pipeline import DocumentProcessingCancelled

    pdf_path = tmp_path / "multi.pdf"
    pdf = pymupdf.open()
    pdf.new_page(width=300, height=200)
    pdf.new_page(width=300, height=200)
    pdf.save(pdf_path)
    pdf.close()

    state = {"calls": 0, "cancel": False}

    class CancellingEngine(FakeEngine):
        def recognize(self, image: Image.Image) -> list[RecognizedLine]:
            state["calls"] += 1
            result = super().recognize(image)
            if state["calls"] == 1:
                state["cancel"] = True
            return result

    with pytest.raises(DocumentProcessingCancelled, match="cancelled"):
        process_document(
            pdf_path,
            engine=CancellingEngine(),
            should_cancel=lambda: state["cancel"],
        )

    assert state["calls"] == 1


def test_image_pixel_limit_is_enforced_before_ocr(tmp_path) -> None:
    import pytest

    from lao_document_ocr.pipeline import DocumentProcessingError

    image_path = tmp_path / "large.png"
    Image.new("RGB", (100, 100), "white").save(image_path)

    with pytest.raises(DocumentProcessingError, match="pixel limit"):
        process_document(
            image_path,
            engine=FakeEngine(),
            max_page_pixels=5_000,
        )


def test_pdf_render_pixel_limit_is_enforced(tmp_path) -> None:
    import pymupdf
    import pytest

    from lao_document_ocr.pipeline import DocumentProcessingError

    pdf_path = tmp_path / "large-page.pdf"
    pdf = pymupdf.open()
    pdf.new_page(width=100, height=100)
    pdf.save(pdf_path)
    pdf.close()

    with pytest.raises(DocumentProcessingError, match="rendered pixel limit"):
        process_document(
            pdf_path,
            engine=FakeEngine(),
            max_page_pixels=30_000,
        )


def test_encrypted_pdf_is_rejected(tmp_path) -> None:
    import pymupdf
    import pytest

    from lao_document_ocr.pipeline import DocumentProcessingError

    pdf_path = tmp_path / "encrypted.pdf"
    pdf = pymupdf.open()
    pdf.new_page(width=100, height=100)
    pdf.save(
        pdf_path,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        user_pw="user-secret",
    )
    pdf.close()

    with pytest.raises(DocumentProcessingError, match="Encrypted PDFs"):
        process_document(pdf_path, engine=FakeEngine())


def test_malformed_image_error_does_not_leak_temp_path(tmp_path) -> None:
    import pytest

    from lao_document_ocr.pipeline import DocumentProcessingError

    image_path = tmp_path / "private-name.png"
    image_path.write_bytes(b"not-an-image")

    with pytest.raises(DocumentProcessingError) as captured:
        process_document(image_path, engine=FakeEngine())

    assert str(captured.value) == "Could not open image."
    assert "private-name" not in str(captured.value)
