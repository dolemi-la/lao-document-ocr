"""Operator-directed rotations are explicit, bounded and never auto-selected."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sys

import pytest
from PIL import Image, ImageOps

import lao_document_ocr.pipeline as pipeline
from lao_document_ocr.conversion import convert_document_to_outputs
from lao_document_ocr.embedded_images import EmbeddedImageAsset
from lao_document_ocr.models import BoundingBox
from lao_document_ocr.ocr.base import OcrEngine, OcrEngineError, RecognizedLine
from lao_document_ocr.page_rotations import parse_page_rotations, validate_page_rotations


class RecordingEngine(OcrEngine):
    def __init__(self, *, vertical=False, after_call=None):
        self.images = []
        self.visual_sources = []
        self.vertical = vertical
        self.after_call = after_call

    def is_available(self):
        return True

    def recognize(self, image):
        self.images.append(image.copy())
        if self.after_call is not None:
            self.after_call()
        return [
            RecognizedLine(
                text="original fixture words",
                bbox=BoundingBox(
                    x=10,
                    y=10 + index * 25,
                    width=12 if self.vertical else 70,
                    height=70 if self.vertical else 12,
                ),
                confidence=0.99,
                block_id=1,
                paragraph_id=1,
                line_id=index + 1,
            )
            for index in range(2)
        ]

    def visual_blocks(self, image, *, source_image=None, exclude_boxes=None):
        self.visual_sources.append((source_image.copy(), list(exclude_boxes)))
        return []


def patterned_image(size=(180, 120)):
    image = Image.new("RGB", size, "white")
    image.paste((220, 10, 70), (10, 10, 24, 22))
    image.paste((0, 80, 220), (size[0] - 26, size[1] - 18, size[0] - 10, size[1] - 7))
    return image


def assert_pixels(actual, expected):
    assert actual.mode == expected.mode
    assert actual.size == expected.size
    assert actual.tobytes() == expected.tobytes()


def test_parser_retains_explicit_zero_and_sorts_pages():
    assert parse_page_rotations(["3:270", "1:0", "2:90"]) == {1: 0, 2: 90, 3: 270}
    assert parse_page_rotations([]) == {}


@pytest.mark.parametrize(
    "spec",
    [
        "",
        "0:90",
        "-1:90",
        "1:-90",
        "1:360",
        "1:45",
        "1:90.0",
        "1.0:90",
        " 1:90",
        "1:90 ",
        "1:090",
        "01:90",
        "1:90:180",
        "x:90",
        "1",
        "١:90",
        None,
    ],
)
def test_parser_rejects_invalid_spec_without_echoing_payload(spec):
    with pytest.raises(ValueError, match="requires PAGE:DEGREES"):
        parse_page_rotations([spec])


@pytest.mark.parametrize("specs", [["1:90", "1:90"], ["1:0", "1:270"]])
def test_parser_rejects_duplicate_pages(specs):
    with pytest.raises(ValueError, match="repeats page 1"):
        parse_page_rotations(specs)


@pytest.mark.parametrize(
    "mapping",
    [
        [],
        "1:90",
        {True: 90},
        {0: 90},
        {-1: 90},
        {"1": 90},
        {1.0: 90},
        {1: True},
        {1: "90"},
        {1: 90.0},
        {1: -90},
        {1: 360},
        {1: None},
    ],
)
def test_mapping_rejects_coercion_and_invalid_angles(mapping):
    with pytest.raises(ValueError):
        validate_page_rotations(mapping)


def test_validation_copies_mapping_and_checks_document_bounds():
    source = {2: 90, 1: 0}
    result = validate_page_rotations(source, page_count=2)
    source[1] = 180
    assert result == {1: 0, 2: 90}
    assert validate_page_rotations(None, page_count=0) == {}
    with pytest.raises(ValueError, match="exceeds the document page count"):
        validate_page_rotations({3: 90}, page_count=2)
    for count in (-1, True, 2.0):
        with pytest.raises(ValueError, match="page_count"):
            validate_page_rotations({}, page_count=count)


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_manual_rotation_precedes_cleanup_and_rotates_preserved_source(
    tmp_path, monkeypatch, angle
):
    image = patterned_image()
    source = tmp_path / "source.png"
    image.save(source)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = pipeline._rotate_image_clockwise(image, angle)
    cleaned_inputs = []

    def cleanup(value):
        cleaned_inputs.append(value.copy())
        return value.convert("L")

    monkeypatch.setattr(pipeline, "preprocess_image", cleanup)
    engine = RecordingEngine()
    document = pipeline.process_document(source, engine=engine, page_rotations={1: angle})
    assert len(engine.images) == 1
    assert_pixels(cleaned_inputs[0], expected)
    assert_pixels(engine.images[0], expected.convert("L"))
    assert_pixels(engine.visual_sources[0][0], expected)
    assert (document.pages[0].width, document.pages[0].height) == expected.size
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    manual = document.metadata["manual_page_rotations"]
    assert manual["pages"] == [{"page": 1, "degrees_clockwise": angle}]
    assert manual["version"] == "explicit-page-rotation-v1"
    assert manual["basis"] == "loaded-page-before-cleanup"
    assert manual["review"]["status"] == "no-sideways-evidence"
    automatic = document.metadata["auto_orientation"]
    assert automatic["enabled"] is False
    assert automatic["pages"][0]["degrees_clockwise"] == 0
    assert automatic["review"]["status"] == "not-requested"
    assert "original fixture" not in json.dumps(manual)


@pytest.mark.parametrize("angle", [0, 90, 270])
def test_manual_override_cannot_be_undone_by_auto_even_with_sideways_output(
    tmp_path,
    monkeypatch,
    angle,
):
    source = tmp_path / "page.png"
    patterned_image().save(source)
    monkeypatch.setattr(pipeline, "preprocess_image", lambda image: image.convert("L"))
    engine = RecordingEngine(vertical=True)
    document = pipeline.process_document(
        source,
        engine=engine,
        page_rotations={1: angle},
        auto_orient_right_angles=True,
    )
    assert len(engine.images) == 1
    automatic = document.metadata["auto_orientation"]
    diagnostics = automatic["pages"][0]["diagnostics"]
    assert diagnostics["probe_skip_reason"] == "manual-override"
    assert diagnostics["probed_degrees"] == []
    assert automatic["pages"][0]["degrees_clockwise"] == 0
    assert automatic["review"]["review_pages"] == [1]
    assert document.metadata["manual_page_rotations"]["review"]["status"] == "review-required"


def test_other_pages_keep_auto_processing_and_selection_is_snapshotted(tmp_path, monkeypatch):
    pages = [pipeline.LoadedPage(patterned_image()) for _ in range(3)]
    monkeypatch.setattr(pipeline, "_load_pages", lambda *args, **kwargs: pages)
    monkeypatch.setattr(pipeline, "preprocess_image", lambda image: image.convert("L"))
    choices = {1: 0, 2: 90}
    engine = RecordingEngine(after_call=lambda: choices.update({2: 270, 3: 180}))
    calls = []

    def automatic(image, **kwargs):
        calls.append(image.copy())
        return image, kwargs["engine"].recognize(image), 0, None

    monkeypatch.setattr(pipeline, "_recognize_with_right_angle_orientation", automatic)
    document = pipeline.process_document(
        tmp_path / "fixture.pdf",
        engine=engine,
        auto_orient_right_angles=True,
        page_rotations=choices,
    )
    assert len(calls) == 1
    assert [image.size for image in engine.images] == [(180, 120), (120, 180), (180, 120)]
    assert document.metadata["manual_page_rotations"]["pages"] == [
        {"page": 1, "degrees_clockwise": 0},
        {"page": 2, "degrees_clockwise": 90},
    ]
    assert document.metadata["manual_page_rotations"]["review"]["unassessed_pages"] == [3]


def test_invalid_rotations_fail_before_loading(tmp_path, monkeypatch):
    def unexpected_load(*args, **kwargs):
        raise AssertionError("invalid rotation must not load input")

    monkeypatch.setattr(pipeline, "_load_pages", unexpected_load)
    with pytest.raises(ValueError, match="positive integers"):
        pipeline.process_document(tmp_path / "fixture.pdf", page_rotations={0: 90})


def test_out_of_range_page_fails_before_ocr_or_output_writes(tmp_path):
    source = tmp_path / "page.png"
    patterned_image().save(source)
    engine = RecordingEngine()
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="page 2 exceeds"):
        convert_document_to_outputs(source, output, engine=engine, page_rotations={2: 90})
    assert engine.images == []
    assert not output.exists()
    output.mkdir()
    existing = output / "page.json"
    existing.write_bytes(b"keep existing export")
    with pytest.raises(ValueError, match="page 2 exceeds"):
        convert_document_to_outputs(source, output, engine=engine, page_rotations={2: 90})
    assert existing.read_bytes() == b"keep existing export"
    assert list(output.iterdir()) == [existing]


@pytest.mark.parametrize("auto", [False, True])
def test_empty_mapping_is_identical_to_no_option(tmp_path, auto):
    source = tmp_path / "page.png"
    patterned_image().save(source)
    a = pipeline.process_document(source, engine=RecordingEngine(), auto_orient_right_angles=auto)
    b = pipeline.process_document(
        source,
        engine=RecordingEngine(),
        auto_orient_right_angles=auto,
        page_rotations={},
    )
    assert a == b
    assert "manual_page_rotations" not in b.metadata


@pytest.mark.parametrize("angle", [90, 180, 270])
def test_embedded_boxes_payload_and_source_rotate_together(tmp_path, monkeypatch, angle):
    from lao_document_ocr.models import BlockType

    page = Image.new("RGB", (180, 120), "white")
    asset_image = patterned_image((80, 50))
    buffer = io.BytesIO()
    asset_image.save(buffer, format="PNG")
    bbox = BoundingBox(x=20, y=30, width=80, height=50)
    page.paste(asset_image, (20, 30))
    asset = EmbeddedImageAsset(bbox=bbox, png_bytes=buffer.getvalue(), width_ratio=80 / 180, xref=7)
    loaded = pipeline.LoadedPage(page, (asset,))
    monkeypatch.setattr(pipeline, "_load_pages", lambda *args, **kwargs: [loaded])
    monkeypatch.setattr(pipeline, "preprocess_image", lambda image: image.convert("L"))
    engine = RecordingEngine()
    document = pipeline.process_document(
        tmp_path / "fixture.pdf", engine=engine, page_rotations={1: angle}
    )
    expected_box = pipeline._rotate_bbox_clockwise(
        bbox,
        page_width=180,
        page_height=120,
        degrees=angle,
    )
    image_blocks = [block for block in document.pages[0].blocks if block.type == BlockType.IMAGE]
    embedded = next(block for block in image_blocks if block.metadata.get("xref") == 7)
    assert embedded.bbox == expected_box
    with Image.open(io.BytesIO(base64.b64decode(embedded.metadata["image_base64"]))) as actual:
        assert_pixels(actual, pipeline._rotate_image_clockwise(asset_image, angle))
    assert expected_box in engine.visual_sources[0][1]
    rotated = pipeline._rotate_embedded_assets(
        (asset,), page_width=180, page_height=120, degrees=angle
    )[0]
    assert rotated.bbox == expected_box
    with Image.open(io.BytesIO(rotated.png_bytes)) as actual:
        assert_pixels(actual, pipeline._rotate_image_clockwise(asset_image, angle))
    assert rotated.width_ratio == expected_box.width / engine.visual_sources[0][0].width
    assert_pixels(engine.visual_sources[0][0], pipeline._rotate_image_clockwise(page, angle))
    assert loaded.embedded_images == (asset,)
    assert_pixels(loaded.image, page)


@pytest.mark.parametrize("manual", [None, {1: 90}])
@pytest.mark.parametrize("exif_orientation", range(1, 9))
def test_exif_display_orientation_applies_once_to_all_pixel_surfaces(
    tmp_path,
    monkeypatch,
    manual,
    exif_orientation,
):
    image = patterned_image()
    exif = Image.Exif()
    exif[274] = exif_orientation
    source = tmp_path / "camera.jpg"
    image.save(source, exif=exif)
    raw_bytes = source.read_bytes()
    with Image.open(source) as stored:
        expected = ImageOps.exif_transpose(stored).convert("RGB")
    if manual:
        expected = pipeline._rotate_image_clockwise(expected, 90)
    monkeypatch.setattr(
        pipeline, "preprocess_image", lambda image: ImageOps.exif_transpose(image).convert("L")
    )
    engine = RecordingEngine()
    pipeline.process_document(source, engine=engine, page_rotations=manual)
    assert_pixels(engine.visual_sources[0][0], expected)
    assert_pixels(engine.images[0], expected.convert("L"))
    assert engine.visual_sources[0][0].getexif().get(274) in (None, 1)
    assert source.read_bytes() == raw_bytes


def test_native_pdf_rotation_and_page_numbering_are_display_relative(tmp_path, monkeypatch):
    import pymupdf

    source = tmp_path / "pages.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page(width=100, height=60)
        page = pdf.new_page(width=100, height=60)
        page.draw_rect(pymupdf.Rect(5, 5, 20, 15), fill=(1, 0, 0))
        page.set_rotation(90)
        pdf.save(source)
    expected = pipeline._load_pages(source, 60, 40_000_000)
    engine = RecordingEngine()
    monkeypatch.setattr(pipeline, "preprocess_image", lambda image: image.convert("L"))
    document = pipeline.process_document(source, engine=engine, page_rotations={2: 270})
    assert_pixels(engine.images[0], expected[0].image.convert("L"))
    assert_pixels(
        engine.images[1], pipeline._rotate_image_clockwise(expected[1].image, 270).convert("L")
    )
    assert [page.number for page in document.pages] == [1, 2]
    assert document.metadata["manual_page_rotations"]["review"]["unassessed_pages"] == [1]


def test_manual_rotation_preserves_cancellation_and_engine_failures(tmp_path, monkeypatch):
    pages = [pipeline.LoadedPage(patterned_image()) for _ in range(2)]
    monkeypatch.setattr(pipeline, "_load_pages", lambda *args, **kwargs: pages)
    engine = RecordingEngine()
    with pytest.raises(pipeline.DocumentProcessingCancelled):
        pipeline.process_document(
            tmp_path / "fixture.pdf",
            engine=engine,
            page_rotations={1: 90},
            should_cancel=lambda: len(engine.images) > 0,
        )
    assert len(engine.images) == 1
    engine = RecordingEngine()

    def fail(image):
        raise OcrEngineError("fixture OCR failure")

    monkeypatch.setattr(engine, "recognize", fail)
    with pytest.raises(pipeline.DocumentProcessingError, match="fixture OCR failure"):
        pipeline.process_document(tmp_path / "fixture.pdf", engine=engine, page_rotations={1: 90})


def test_cli_manual_rotation_generates_exports_and_provenance(tmp_path, monkeypatch, capsys):
    from lao_document_ocr import cli

    source = tmp_path / "original.png"
    patterned_image().save(source)
    output = tmp_path / "converted"
    engine = RecordingEngine()
    monkeypatch.setattr(cli, "_build_ocr_engine", lambda args: engine)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "convert-document",
            "--input",
            str(source),
            "--output-dir",
            str(output),
            "--rotate-page",
            "1:270",
            "--auto-orient-right-angles",
        ],
    )
    assert cli.main() == 0
    emitted = json.loads(capsys.readouterr().out)
    assert set(emitted) == {"docx", "markdown", "text", "json"}
    payload = json.loads((output / "original.json").read_text())
    assert payload["metadata"]["manual_page_rotations"]["pages"] == [
        {"page": 1, "degrees_clockwise": 270}
    ]
    assert (payload["pages"][0]["width"], payload["pages"][0]["height"]) == (120, 180)
    assert len(engine.images) == 1
    assert all((output / f"original.{ext}").is_file() for ext in ("docx", "md", "txt", "json"))


def test_cli_bad_spec_fails_before_engine_or_output(tmp_path, monkeypatch, capsys):
    from lao_document_ocr import cli

    def unexpected_engine(args):
        raise AssertionError("invalid syntax must fail before engine creation")

    monkeypatch.setattr(cli, "_build_ocr_engine", unexpected_engine)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "convert-document",
            "--input",
            str(tmp_path / "input.png"),
            "--output-dir",
            str(tmp_path / "output"),
            "--rotate-page",
            "1:90",
            "--rotate-page",
            "1:0",
        ],
    )
    assert cli.main() == 1
    assert "repeats page 1" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()
