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


def test_custom_reading_order_resolver_controls_final_page_order(tmp_path) -> None:
    class TwoBlockEngine(OcrEngine):
        def is_available(self) -> bool:
            return True

        def recognize(self, image: Image.Image) -> list[RecognizedLine]:
            return [
                RecognizedLine(
                    text="First",
                    bbox=BoundingBox(x=20, y=20, width=100, height=20),
                    confidence=0.99,
                    block_id=1,
                    paragraph_id=1,
                    line_id=1,
                ),
                RecognizedLine(
                    text="Second",
                    bbox=BoundingBox(x=20, y=80, width=100, height=20),
                    confidence=0.99,
                    block_id=2,
                    paragraph_id=2,
                    line_id=1,
                ),
            ]

    class ReverseResolver:
        def metadata(self):
            return {"name": "ReverseResolver", "version": "test-v1"}

        def order(self, blocks, *, page_width, page_height):
            assert page_width == 320
            assert page_height == 180
            return list(reversed(blocks))

    image_path = tmp_path / "ordered.png"
    Image.new("RGB", (320, 180), "white").save(image_path)

    document = process_document(
        image_path,
        engine=TwoBlockEngine(),
        reading_order_resolver=ReverseResolver(),
    )

    assert [block.text for block in document.pages[0].blocks] == [
        "Second",
        "First",
    ]
    assert document.metadata["reading_order"] == {
        "name": "ReverseResolver",
        "version": "test-v1",
    }


def test_default_reading_order_metadata_is_recorded(tmp_path) -> None:
    image_path = tmp_path / "default-order.png"
    Image.new("RGB", (320, 180), "white").save(image_path)

    document = process_document(image_path, engine=FakeEngine())

    assert document.metadata["reading_order"]["name"] == (
        "DeterministicReadingOrderResolver"
    )
    assert document.metadata["reading_order"]["version"] == "multi-column-v2"


def test_engine_learned_visual_blocks_are_preserved_without_heuristic_duplicate(
    tmp_path,
) -> None:
    import base64
    import io

    from lao_document_ocr.models import Block, BlockType

    class VisualEngine(FakeEngine):
        def recognize(self, image: Image.Image) -> list[RecognizedLine]:
            return []

        def visual_blocks(
            self,
            image: Image.Image,
            *,
            source_image=None,
            exclude_boxes=None,
        ):
            crop = Image.new("RGB", (80, 60), (80, 140, 210))
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG")
            return [
                Block(
                    type=BlockType.IMAGE,
                    bbox=BoundingBox(x=100, y=80, width=80, height=60),
                    metadata={
                        "source": "learned-layout-image",
                        "detector": "tiny-layout-unet-v1",
                        "media_type": "image/png",
                        "image_base64": base64.b64encode(
                            buffer.getvalue()
                        ).decode("ascii"),
                        "width_ratio": 0.25,
                        "area_ratio": 0.08,
                    },
                )
            ]

    image_path = tmp_path / "visual.png"
    page = Image.new("RGB", (400, 300), "white")
    page.paste(Image.new("RGB", (80, 60), (80, 140, 210)), (100, 80))
    page.save(image_path)

    document = process_document(image_path, engine=VisualEngine())

    image_blocks = [
        block
        for block in document.pages[0].blocks
        if block.type == BlockType.IMAGE
    ]
    assert len(image_blocks) == 1
    assert image_blocks[0].metadata["source"] == "learned-layout-image"
