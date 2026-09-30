"""Confident OCR must not suppress an opt-in probe of a sideways baseline."""

from __future__ import annotations

import json

import pytest
from PIL import Image

import lao_document_ocr.pipeline as pipeline
from lao_document_ocr.models import BoundingBox
from lao_document_ocr.ocr.base import OcrEngine, OcrEngineError, RecognizedLine


def make_lines(confidence, *, vertical=False, text="private fixture words " * 4, count=2):
    return [
        RecognizedLine(
            text=text,
            bbox=BoundingBox(
                x=10 + index * 25,
                y=10,
                width=20 if vertical else 250,
                height=250 if vertical else 20,
            ),
            confidence=confidence,
            block_id=1,
            paragraph_id=1,
            line_id=index + 1,
        )
        for index in range(count)
    ]


class SequenceEngine(OcrEngine):
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    def is_available(self):
        return True

    def recognize(self, image):
        self.calls += 1
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return value


@pytest.mark.parametrize(
    "confidence,text,skip_reason",
    [
        (0.70, "private fixture words " * 4, "high-confidence"),
        (0.50, "ລາວ" * 50, "lao-dominant-baseline"),
    ],
    ids=["high-confidence", "lao-dominant"],
)
def test_sideways_baseline_overrides_both_confidence_shortcuts(confidence, text, skip_reason):
    baseline = make_lines(confidence, vertical=True, text=text)
    engine = SequenceEngine(
        [baseline, make_lines(0.95, text=text), make_lines(0.99, vertical=True, text=text), []]
    )
    image = Image.new("RGB", (400, 600), "white")
    rotated, _, degrees, info = pipeline._recognize_with_right_angle_orientation(
        image, engine=engine
    )
    assert degrees == 90
    assert rotated.size == (600, 400)
    assert engine.calls == 4
    assert info["probe_skipped"] is False
    assert info["probe_skip_reason"] is None
    assert info["probe_skip_override_reason"] == "sideways-baseline"
    assert info["probe_skip_policy"] == "confidence-with-line-axis-v1"
    assert info["geometry_vetoed_degrees"] == [180]
    assert info["candidate_geometry"][0]["geometry"]["sideways_dominant"]

    # Same confidence and labels with horizontal geometry retain the old shortcut.
    horizontal = SequenceEngine([make_lines(confidence, text=text)])
    _, _, degrees, info = pipeline._recognize_with_right_angle_orientation(image, engine=horizontal)
    assert degrees == 0
    assert horizontal.calls == 1
    assert info["probe_skip_reason"] == skip_reason
    assert info["probe_skip_override_reason"] is None


@pytest.mark.parametrize("confidence", [0.65, 0.80, 0.97])
def test_sideways_override_does_not_relax_any_acceptance_threshold(confidence):
    baseline = make_lines(confidence, vertical=True)
    engine = SequenceEngine(
        [baseline, make_lines(confidence + 0.02), make_lines(0.99, vertical=True), []]
    )
    image = Image.new("RGB", (400, 600), "white")
    selected_image, selected_lines, degrees, info = (
        pipeline._recognize_with_right_angle_orientation(image, engine=engine)
    )
    assert degrees == 0
    assert selected_image is image
    assert selected_lines is baseline
    assert info["probe_skipped"] is False
    assert info["probe_skip_override_reason"] == "sideways-baseline"
    assert engine.calls == 4


@pytest.mark.parametrize("kind", ["one-vertical", "single-character", "square", "equal-support"])
def test_ambiguous_axis_keeps_high_confidence_fast_path(kind):
    if kind == "one-vertical":
        baseline = make_lines(0.95, vertical=True, count=1)
    elif kind == "single-character":
        baseline = make_lines(0.95, vertical=True, text="x")
    elif kind == "square":
        baseline = [
            RecognizedLine(
                "private fixture", BoundingBox(x=1, y=1, width=20, height=20), 0.95, 1, 1, index
            )
            for index in range(2)
        ]
    else:
        baseline = make_lines(0.95, vertical=True) + make_lines(0.95)
    engine = SequenceEngine([baseline])
    image = Image.new("RGB", (400, 600), "white")
    selected, _, degrees, info = pipeline._recognize_with_right_angle_orientation(
        image, engine=engine
    )
    assert selected is image and degrees == 0
    assert engine.calls == 1
    assert info["probe_skip_reason"] == "high-confidence"
    assert info["probe_skip_override_reason"] is None


def test_low_confidence_sideways_probe_is_not_reported_as_shortcut_override():
    engine = SequenceEngine([make_lines(0.30, vertical=True), make_lines(0.95), [], []])
    _, _, degrees, info = pipeline._recognize_with_right_angle_orientation(
        Image.new("RGB", (400, 600)), engine=engine
    )
    assert degrees == 90
    assert engine.calls == 4
    assert info["probe_skip_override_reason"] is None


def test_all_failed_probes_keep_confident_sideways_baseline_and_hide_errors():
    baseline = make_lines(0.75, vertical=True)
    engine = SequenceEngine([baseline, *[OcrEngineError("PRIVATE-ERROR") for _ in range(3)]])
    image = Image.new("RGB", (400, 600), "white")
    selected, selected_lines, degrees, info = pipeline._recognize_with_right_angle_orientation(
        image, engine=engine
    )
    assert selected is image and selected_lines is baseline and degrees == 0
    assert engine.calls == 4
    assert info["probed_degrees"] == []
    assert len(info["candidate_geometry"]) == 1
    assert info["runner_up_score"] is None
    encoded = json.dumps(info, allow_nan=False)
    assert "PRIVATE-ERROR" not in encoded
    assert "private fixture words" not in encoded
    assert "bbox" not in encoded


def test_cancel_before_geometry_triggered_probe_does_not_call_engine_again():
    engine = SequenceEngine([make_lines(0.75, vertical=True)])
    with pytest.raises(pipeline.DocumentProcessingCancelled):
        pipeline._recognize_with_right_angle_orientation(
            Image.new("RGB", (400, 600)), engine=engine, should_cancel=lambda: True
        )
    assert engine.calls == 1


@pytest.mark.parametrize("auto", [False, True])
def test_document_path_keeps_default_off_and_rotates_only_accepted_opt_in(tmp_path, auto):
    path = tmp_path / "own-fixture.png"
    Image.new("RGB", (400, 600), "white").save(path)
    engine = SequenceEngine(
        [make_lines(0.70, vertical=True), make_lines(0.95), make_lines(0.99, vertical=True), []]
    )
    document = pipeline.process_document(path, engine=engine, auto_orient_right_angles=auto)
    assert engine.calls == (4 if auto else 1)
    assert document.metadata["auto_orientation"]["enabled"] is auto
    assert document.metadata["auto_orientation"]["pages"][0]["degrees_clockwise"] == (
        90 if auto else 0
    )
    assert (document.pages[0].width, document.pages[0].height) == (
        (600, 400) if auto else (400, 600)
    )


@pytest.mark.parametrize(
    "baseline_confidence,baseline_text,candidate_confidence,candidate_text",
    [(0.70, "a" * 40, 0.80, "a" * 40), (0.45, "ລາວ" * 40, 0.99, "ລາວ" * 19)],
    ids=["score-improvement-still-required", "character-retention-still-required"],
)
def test_geometry_probe_preserves_score_and_character_thresholds(
    baseline_confidence, baseline_text, candidate_confidence, candidate_text
):
    baseline = make_lines(baseline_confidence, vertical=True, text=baseline_text)
    candidate = make_lines(candidate_confidence, text=candidate_text)
    engine = SequenceEngine([baseline, candidate, [], []])
    image = Image.new("RGB", (400, 600), "white")
    selected, selected_lines, degrees, info = pipeline._recognize_with_right_angle_orientation(
        image, engine=engine
    )
    assert selected is image and selected_lines is baseline and degrees == 0
    assert info["probe_skip_override_reason"] == "sideways-baseline"
    assert info["best_degrees_clockwise"] == 90
    assert engine.calls == 4
