"""Orientation confidence must not favor clearly sideways multi-character lines."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from PIL import Image

import lao_document_ocr.pipeline as pipeline
from lao_document_ocr.models import BoundingBox
from lao_document_ocr.ocr.base import OcrEngine, OcrEngineError, RecognizedLine
from lao_document_ocr.orientation_geometry import orientation_line_geometry


def line(text="PRIVATE-LABEL-ABC", width=100, height=20, confidence=0.90):
    return RecognizedLine(
        text, BoundingBox(x=10, y=10, width=width, height=height), confidence, 1, 1, 1
    )


def lines(confidence, *, vertical=False, count=2, length=40):
    return [
        line(
            "x" * length,
            width=20 if vertical else 100,
            height=100 if vertical else 20,
            confidence=confidence,
        )
        for _ in range(count)
    ]


def test_vertical_evidence_requires_repeated_multi_character_lines():
    assert not orientation_line_geometry(lines(0.95, vertical=True, count=1))["sideways_dominant"]
    report = orientation_line_geometry(lines(0.95, vertical=True))
    assert report["sideways_dominant"]
    assert report["vertical_lines"] == 2
    assert report["vertical_characters"] == 80
    assert report["horizontal_characters"] == 0


def test_evidence_is_character_weighted_not_line_vote():
    report = orientation_line_geometry(
        [
            *lines(0.95, vertical=True, count=3, length=2),
            line("horizontal characters dominate"),
        ]
    )
    assert not report["sideways_dominant"]
    assert report["vertical_lines"] == 3
    assert report["vertical_characters"] == 6


def test_equal_axis_support_and_noninformative_boxes_do_not_veto():
    report = orientation_line_geometry(
        [
            *lines(0.9, vertical=True, count=2, length=3),
            line("abcdef"),
            line("x", width=2, height=20),
            line("square label", width=20, height=20),
            line("  \t "),
        ]
    )
    assert not report["sideways_dominant"]
    assert report["ignored_lines"] == 3


@pytest.mark.parametrize(
    "width,height", [(0, 10), (-1, 10), (10, 0), (float("nan"), 10), (10, float("inf"))]
)
def test_invalid_geometry_is_unknown_not_sideways(width, height):
    fake = SimpleNamespace(text="private label", bbox=SimpleNamespace(width=width, height=height))
    report = orientation_line_geometry([fake])
    assert report["ignored_lines"] == 1
    assert not report["sideways_dominant"]


def test_geometry_report_exposes_no_text_ids_or_boxes():
    report = orientation_line_geometry([line(), line(width=20, height=100)])
    assert set(report) == {
        "version",
        "horizontal_lines",
        "vertical_lines",
        "horizontal_characters",
        "vertical_characters",
        "ignored_lines",
        "sideways_dominant",
    }
    assert "PRIVATE-LABEL" not in json.dumps(report, allow_nan=False)


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


def test_sideways_high_score_is_vetoed_before_ranking():
    engine = SequenceEngine([lines(0.3), lines(0.90), lines(0.99, vertical=True), lines(0.2)])
    image = Image.new("RGB", (400, 600), "white")
    rotated, _, angle, diagnostics = pipeline._recognize_with_right_angle_orientation(
        image, engine=engine
    )
    assert angle == 90
    assert rotated.size == (600, 400)
    assert engine.calls == 4
    assert diagnostics["geometry_vetoed_degrees"] == [180]
    assert diagnostics["best_degrees_clockwise"] == 90
    assert len(diagnostics["candidate_geometry"]) == 4
    assert diagnostics["candidate_geometry"][2]["geometry"]["sideways_dominant"]


def test_veto_does_not_relax_existing_improvement_thresholds():
    engine = SequenceEngine([lines(0.3), lines(0.31), lines(0.99, vertical=True), lines(0.32)])
    image = Image.new("RGB", (400, 600), "white")
    result, _, angle, _ = pipeline._recognize_with_right_angle_orientation(image, engine=engine)
    assert result is image
    assert angle == 0
    assert engine.calls == 4


def test_single_tall_label_does_not_override_previous_ranking():
    engine = SequenceEngine(
        [lines(0.3), lines(0.9), lines(0.99, vertical=True, count=1, length=80), lines(0.2)]
    )
    _, _, angle, _ = pipeline._recognize_with_right_angle_orientation(
        Image.new("RGB", (400, 600)), engine=engine
    )
    assert angle == 180


def test_failed_probe_has_no_geometry_and_no_exception_text():
    engine = SequenceEngine([lines(0.3), lines(0.9), OcrEngineError("PRIVATE-ERROR"), lines(0.2)])
    _, _, angle, info = pipeline._recognize_with_right_angle_orientation(
        Image.new("RGB", (400, 600)), engine=engine
    )
    assert angle == 90
    assert len(info["candidate_geometry"]) == 3
    assert "PRIVATE-ERROR" not in json.dumps(info)
    assert engine.calls == 4


def test_high_confidence_horizontal_skip_is_unchanged():
    engine = SequenceEngine([lines(0.95)])
    _, _, angle, info = pipeline._recognize_with_right_angle_orientation(
        Image.new("RGB", (400, 600)), engine=engine
    )
    assert angle == 0
    assert info["probe_skip_reason"] == "high-confidence"
    assert engine.calls == 1


def test_opt_in_pipeline_uses_geometry_but_default_never_probes(tmp_path):
    path = tmp_path / "own-blank-fixture.png"
    Image.new("RGB", (400, 600), "white").save(path)
    default_engine = SequenceEngine([lines(0.3)])
    document = pipeline.process_document(path, engine=default_engine)
    assert default_engine.calls == 1
    assert not document.metadata["auto_orientation"]["enabled"]
    engine = SequenceEngine([lines(0.3), lines(0.9), lines(0.99, vertical=True), lines(0.2)])
    document = pipeline.process_document(
        path, engine=engine, auto_orient_right_angles=True, include_ocr_line_stats=True
    )
    assert document.metadata["auto_orientation"]["pages"][0]["degrees_clockwise"] == 90
    assert (document.pages[0].width, document.pages[0].height) == (600, 400)
    assert not document.metadata["ocr_line_stats"]["pages"][0]["orientation_geometry"][
        "sideways_dominant"
    ]


@pytest.mark.parametrize("has_geometry", [True, False])
def test_standalone_diagnostic_uses_same_geometry_filter(tmp_path, monkeypatch, has_geometry):
    import lao_document_ocr.remote_evaluation as remote

    def stats(outputs):
        confidence, count, score, ratio = pipeline._orientation_line_stats(outputs)
        line_stats = {"mean_confidence": confidence, "recognized_characters": count, "score": score}
        if has_geometry:
            line_stats["orientation_geometry"] = orientation_line_geometry(outputs)
        return {
            "line_stats": line_stats,
            "mean_block_confidence": confidence,
            "text": {"letter_characters": count},
        }

    baseline = stats(lines(0.3))
    candidates = iter([stats(lines(0.9)), stats(lines(0.99, vertical=True)), stats(lines(0.2))])
    monkeypatch.setattr(remote, "process_document", lambda *args, **kwargs: next(candidates))
    monkeypatch.setattr(remote, "_ocr_document_stats", lambda value: value)
    source = tmp_path / "own-fixture.png"
    Image.new("RGB", (400, 600), "white").save(source)
    report = remote._rotation_probe(
        source,
        baseline_ocr=baseline,
        engine=SequenceEngine([]),
        max_page_pixels=1_000_000,
        reading_order_resolver=None,
    )
    assert report["recommended_degrees_clockwise"] == (90 if has_geometry else 180)
    assert report["geometry_vetoed_degrees"] == ([180] if has_geometry else [])
    assert len(report["variants"]) == 4
    assert not list(tmp_path.glob("*rot*.png"))


def test_remote_geometry_allowlist_drops_arbitrary_fields_and_untrusted_flag():
    from lao_document_ocr.orientation_geometry import sanitized_orientation_geometry

    report = orientation_line_geometry(lines(0.9))
    payload = {**report, "private": "PRIVATE-TEXT", "sideways_dominant": True}
    assert sanitized_orientation_geometry(payload) == report
    assert sanitized_orientation_geometry({**report, "version": "unknown"}) is None
    assert sanitized_orientation_geometry({**report, "vertical_lines": "PRIVATE-TEXT"}) is None
    assert sanitized_orientation_geometry({**report, "vertical_lines": True}) is None
    assert sanitized_orientation_geometry({**report, "horizontal_characters": -1}) is None
    assert sanitized_orientation_geometry(None) is None


def test_all_rotated_candidates_vetoed_keep_baseline():
    engine = SequenceEngine(
        [
            lines(0.3),
            lines(0.9, vertical=True),
            lines(0.99, vertical=True),
            lines(0.95, vertical=True),
        ]
    )
    image = Image.new("RGB", (400, 600), "white")
    result, _, angle, info = pipeline._recognize_with_right_angle_orientation(image, engine=engine)
    assert result is image
    assert angle == 0
    assert info["geometry_vetoed_degrees"] == [90, 180, 270]
    assert info["runner_up_degrees_clockwise"] is None
    assert info["best_score_margin_ratio"] is None
    assert engine.calls == 4
