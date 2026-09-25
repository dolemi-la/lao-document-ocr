import io

from PIL import Image

from lao_document_ocr.models import BoundingBox
from lao_document_ocr.ocr.base import OcrEngine, OcrEngineError, RecognizedLine
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
    assert "ocr_line_stats" not in document.metadata


def test_process_document_can_include_safe_ocr_line_stats(tmp_path) -> None:
    image_path = tmp_path / "sample-line-stats.png"
    Image.new("RGB", (320, 180), "white").save(image_path)

    document = process_document(
        image_path,
        engine=FakeEngine(),
        include_ocr_line_stats=True,
    )

    stats = document.metadata["ocr_line_stats"]["pages"][0]
    assert stats["page"] == 1
    assert stats["mean_confidence"] == 0.99
    assert stats["recognized_characters"] == 8
    assert stats["score"] > 0
    assert stats["lao_ratio"] == 0.0
    assert "text" not in stats


def test_safe_ocr_line_stats_ignore_whitespace_only_lines(tmp_path) -> None:
    image_path = tmp_path / "sample-whitespace-line-stats.png"
    Image.new("RGB", (320, 180), "white").save(image_path)

    class WhitespaceEngine(FakeEngine):
        def recognize(self, image: Image.Image) -> list[RecognizedLine]:
            return [
                RecognizedLine(
                    text="   \n\t",
                    bbox=BoundingBox(x=10, y=10, width=120, height=24),
                    confidence=0.99,
                    block_id=1,
                    paragraph_id=1,
                    line_id=1,
                )
            ]

    document = process_document(
        image_path,
        engine=WhitespaceEngine(),
        include_ocr_line_stats=True,
    )

    stats = document.metadata["ocr_line_stats"]["pages"][0]
    assert stats["mean_confidence"] == 0.0
    assert stats["recognized_characters"] == 0
    assert stats["score"] == 0.0
    assert stats["lao_ratio"] == 0.0


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


class OrientationSensitiveEngine(OcrEngine):
    def is_available(self) -> bool:
        return True

    def recognize(self, image: Image.Image) -> list[RecognizedLine]:
        confidence = 0.95 if image.height > image.width else 0.30
        return [
            RecognizedLine(
                text="orientation text with enough recognized characters",
                bbox=BoundingBox(x=10, y=10, width=100, height=20),
                confidence=confidence,
                block_id=1,
                paragraph_id=1,
                line_id=1,
            )
        ]


def test_auto_orientation_rotates_page_and_records_metadata(tmp_path) -> None:
    image_path = tmp_path / "landscape.png"
    Image.new("RGB", (320, 180), "white").save(image_path)

    document = process_document(
        image_path,
        engine=OrientationSensitiveEngine(),
        auto_orient_right_angles=True,
    )

    page = document.pages[0]
    assert (page.width, page.height) == (180, 320)
    assert document.metadata["auto_orientation"]["enabled"] is True
    orientation = document.metadata["auto_orientation"]["pages"][0]
    assert orientation["degrees_clockwise"] == 90
    assert orientation["diagnostics"]["selected_confidence"] > 0.9
    assert orientation["diagnostics"]["baseline_confidence"] < 0.4
    assert orientation["diagnostics"]["probe_skipped"] is False
    assert orientation["diagnostics"]["best_degrees_clockwise"] == 90
    assert orientation["diagnostics"]["runner_up_degrees_clockwise"] == 270
    assert orientation["diagnostics"]["best_score_margin_ratio"] == 0.0


def test_auto_orientation_is_disabled_by_default(tmp_path) -> None:
    image_path = tmp_path / "landscape-default.png"
    Image.new("RGB", (320, 180), "white").save(image_path)

    document = process_document(image_path, engine=OrientationSensitiveEngine())

    page = document.pages[0]
    assert (page.width, page.height) == (320, 180)
    assert document.metadata["auto_orientation"]["enabled"] is False
    assert document.metadata["auto_orientation"]["pages"][0][
        "degrees_clockwise"
    ] == 0


def test_auto_orientation_keeps_baseline_without_clear_improvement(tmp_path) -> None:
    state = {"calls": 0}

    class StableEngine(OcrEngine):
        def is_available(self) -> bool:
            return True

        def recognize(self, image: Image.Image) -> list[RecognizedLine]:
            state["calls"] += 1
            return [
                RecognizedLine(
                    text="stable orientation recognition text",
                    bbox=BoundingBox(x=10, y=10, width=100, height=20),
                    confidence=0.85,
                    block_id=1,
                    paragraph_id=1,
                    line_id=1,
                )
            ]

    image_path = tmp_path / "stable.png"
    Image.new("RGB", (320, 180), "white").save(image_path)

    document = process_document(
        image_path,
        engine=StableEngine(),
        auto_orient_right_angles=True,
    )

    assert (document.pages[0].width, document.pages[0].height) == (320, 180)
    assert document.metadata["auto_orientation"]["pages"][0][
        "degrees_clockwise"
    ] == 0
    diagnostics = document.metadata["auto_orientation"]["pages"][0]["diagnostics"]
    assert diagnostics["probe_skipped"] is True
    assert diagnostics["probe_below_confidence"] == 0.65
    assert state["calls"] == 1


def test_auto_orientation_honors_cancellation_between_probes(tmp_path) -> None:
    import pytest

    from lao_document_ocr.pipeline import DocumentProcessingCancelled

    state = {"calls": 0, "cancel": False}

    class CancellingOrientationEngine(OrientationSensitiveEngine):
        def recognize(self, image: Image.Image) -> list[RecognizedLine]:
            state["calls"] += 1
            lines = super().recognize(image)
            if state["calls"] == 1:
                state["cancel"] = True
            return lines

    image_path = tmp_path / "cancel-orientation.png"
    Image.new("RGB", (320, 180), "white").save(image_path)

    with pytest.raises(DocumentProcessingCancelled):
        process_document(
            image_path,
            engine=CancellingOrientationEngine(),
            auto_orient_right_angles=True,
            should_cancel=lambda: state["cancel"],
        )

    assert state["calls"] == 1


def test_right_angle_bbox_rotation_coordinates() -> None:
    from lao_document_ocr.pipeline import _rotate_bbox_clockwise

    bbox = BoundingBox(x=10, y=20, width=30, height=40)

    rotated_90 = _rotate_bbox_clockwise(
        bbox,
        page_width=200,
        page_height=100,
        degrees=90,
    )
    assert rotated_90 == BoundingBox(x=40, y=10, width=40, height=30)

    rotated_180 = _rotate_bbox_clockwise(
        bbox,
        page_width=200,
        page_height=100,
        degrees=180,
    )
    assert rotated_180 == BoundingBox(x=160, y=40, width=30, height=40)

    rotated_270 = _rotate_bbox_clockwise(
        bbox,
        page_width=200,
        page_height=100,
        degrees=270,
    )
    assert rotated_270 == BoundingBox(x=20, y=160, width=40, height=30)


def test_embedded_asset_bbox_rotates_with_page() -> None:
    from lao_document_ocr.embedded_images import EmbeddedImageAsset
    from lao_document_ocr.pipeline import _rotate_embedded_assets

    source = Image.new("RGB", (2, 3), "black")
    source.putpixel((0, 0), (255, 0, 0))
    source.putpixel((1, 0), (0, 255, 0))
    source.putpixel((0, 1), (0, 0, 255))
    source.putpixel((1, 1), (255, 255, 0))
    source.putpixel((0, 2), (255, 0, 255))
    source.putpixel((1, 2), (0, 255, 255))
    payload = io.BytesIO()
    source.save(payload, format="PNG")

    asset = EmbeddedImageAsset(
        bbox=BoundingBox(x=10, y=20, width=30, height=40),
        png_bytes=payload.getvalue(),
        width_ratio=0.15,
        xref=7,
    )

    rotated = _rotate_embedded_assets(
        (asset,),
        page_width=200,
        page_height=100,
        degrees=90,
    )

    assert len(rotated) == 1
    assert rotated[0].bbox == BoundingBox(x=40, y=10, width=40, height=30)
    assert rotated[0].width_ratio == 0.4
    assert rotated[0].xref == 7
    with Image.open(io.BytesIO(rotated[0].png_bytes)) as rotated_image:
        assert rotated_image.size == (3, 2)
        assert rotated_image.getpixel((0, 0)) == (255, 0, 255)
        assert rotated_image.getpixel((2, 0)) == (255, 0, 0)


def test_auto_orientation_skips_probe_for_strong_lao_baseline(tmp_path) -> None:
    state = {"calls": 0}

    class LaoDominantEngine(OcrEngine):
        def is_available(self) -> bool:
            return True

        def recognize(self, image: Image.Image) -> list[RecognizedLine]:
            state["calls"] += 1
            return [
                RecognizedLine(
                    text="ສະບາຍດີ" * 40,
                    bbox=BoundingBox(x=10, y=10, width=200, height=30),
                    confidence=0.50,
                    block_id=1,
                    paragraph_id=1,
                    line_id=1,
                )
            ]

    image_path = tmp_path / "lao-dominant.png"
    Image.new("RGB", (320, 180), "white").save(image_path)

    document = process_document(
        image_path,
        engine=LaoDominantEngine(),
        auto_orient_right_angles=True,
    )

    orientation = document.metadata["auto_orientation"]["pages"][0]
    assert orientation["degrees_clockwise"] == 0
    assert orientation["diagnostics"]["probe_skipped"] is True
    assert orientation["diagnostics"]["probe_skip_reason"] == (
        "lao-dominant-baseline"
    )
    assert orientation["diagnostics"]["baseline_lao_ratio"] >= 0.75
    assert orientation["diagnostics"]["best_score_margin_ratio"] is None
    assert state["calls"] == 1


def test_auto_orientation_ignores_engine_hint_and_selects_best_exhaustive_candidate(
    tmp_path,
) -> None:
    state = {"recognize_calls": 0, "hint_calls": 0}

    class HintedEngine(OcrEngine):
        def is_available(self) -> bool:
            return True

        def orientation_hint(self, image: Image.Image):
            del image
            state["hint_calls"] += 1
            return {
                "degrees_clockwise": 90,
                "orientation_confidence": 20.0,
                "source": "test-hint",
            }

        def recognize(self, image: Image.Image) -> list[RecognizedLine]:
            del image
            state["recognize_calls"] += 1
            confidence_by_call = {
                1: 0.30,  # baseline
                2: 0.82,  # 90°: acceptable, but not best
                3: 0.95,  # 180°: best
                4: 0.40,  # 270°
            }
            confidence = confidence_by_call[state["recognize_calls"]]
            return [
                RecognizedLine(
                    text="exhaustive orientation text with enough characters",
                    bbox=BoundingBox(x=10, y=10, width=100, height=20),
                    confidence=confidence,
                    block_id=1,
                    paragraph_id=1,
                    line_id=1,
                )
            ]

    image_path = tmp_path / "hinted.png"
    Image.new("RGB", (320, 180), "white").save(image_path)

    document = process_document(
        image_path,
        engine=HintedEngine(),
        auto_orient_right_angles=True,
    )

    orientation = document.metadata["auto_orientation"]["pages"][0]
    assert orientation["degrees_clockwise"] == 180
    diagnostics = orientation["diagnostics"]
    assert diagnostics["probe_strategy"] == "exhaustive"
    assert diagnostics["probed_degrees"] == [90, 180, 270]
    assert diagnostics["engine_orientation_hint"] is None
    assert diagnostics["best_degrees_clockwise"] == 180
    assert diagnostics["runner_up_degrees_clockwise"] == 90
    assert diagnostics["best_score_margin_ratio"] > 0.0
    assert state["hint_calls"] == 0
    assert state["recognize_calls"] == 4


def test_auto_orientation_exhaustive_probe_continues_after_candidate_error(
    tmp_path,
) -> None:
    state = {"calls": 0}

    class PartialFailureEngine(OcrEngine):
        def is_available(self) -> bool:
            return True

        def recognize(self, image: Image.Image) -> list[RecognizedLine]:
            del image
            state["calls"] += 1
            if state["calls"] == 2:
                raise OcrEngineError("90-degree probe failed")
            confidence_by_call = {
                1: 0.30,  # baseline
                3: 0.95,  # 180°
                4: 0.40,  # 270°
            }
            confidence = confidence_by_call[state["calls"]]
            return [
                RecognizedLine(
                    text="fallback orientation text with enough characters",
                    bbox=BoundingBox(x=10, y=10, width=100, height=20),
                    confidence=confidence,
                    block_id=1,
                    paragraph_id=1,
                    line_id=1,
                )
            ]

    image_path = tmp_path / "partial-probe-failure.png"
    Image.new("RGB", (320, 180), "white").save(image_path)

    document = process_document(
        image_path,
        engine=PartialFailureEngine(),
        auto_orient_right_angles=True,
    )

    orientation = document.metadata["auto_orientation"]["pages"][0]
    assert orientation["degrees_clockwise"] == 180
    diagnostics = orientation["diagnostics"]
    assert diagnostics["probe_strategy"] == "exhaustive"
    assert diagnostics["probed_degrees"] == [180, 270]
    assert state["calls"] == 4
