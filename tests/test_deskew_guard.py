"""Rights-clear geometric fixtures for conservative small-angle deskew evidence."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image, ImageEnhance, ImageOps

import lao_document_ocr.preprocessing as preprocessing


def _rows():
    image = np.full((300, 1000), 255, dtype=np.uint8)
    for y in (85, 145, 205):
        for x in range(150, 850, 30):
            cv2.rectangle(image, (x, y - 8), (x + 15, y + 8), 0, -1)
    return image


def _mask(image):
    return (image < 128).astype(np.uint8) * 255


@pytest.mark.parametrize("outlier", [(20, 20), (100, 20), (900, 20), (940, 250)])
def test_upright_rows_with_corner_graphic_do_not_follow_foreground_angle(outlier):
    image = _rows()
    cv2.circle(image, outlier, 12, 0, -1)
    points = cv2.findNonZero(_mask(image))
    angle = (float(cv2.minAreaRect(points)[-1]) + 45) % 90 - 45
    assert 0.15 < abs(angle) < 12  # The old whole-foreground gate would rotate.
    original = image.copy()
    corrected = preprocessing._deskew(image)
    np.testing.assert_array_equal(corrected, original)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("tilt", [-6.0, -3.0, 3.0, 6.0])
def test_supported_correction_improves_alignment_without_dropping_foreground(tilt):
    tilted = cv2.warpAffine(
        _rows(),
        cv2.getRotationMatrix2D((500, 150), tilt, 1),
        (1000, 300),
        borderValue=255,
    )
    mask = _mask(tilted)
    points = cv2.findNonZero(mask)
    correction = (float(cv2.minAreaRect(points)[-1]) + 45) % 90 - 45
    evidence = preprocessing._deskew_evidence(
        mask,
        points,
        cv2.getRotationMatrix2D((500, 150), correction, 1),
    )
    assert evidence["accepted"] is True
    assert evidence["reason"] == "alignment-improved"
    assert evidence["candidate_score"] > evidence["baseline_score"]
    assert evidence["foreground_within_canvas"] is True


def test_unsupported_alignment_returns_original_pixels_even_with_safe_bounds():
    image = _rows()
    mask = _mask(image)
    evidence = preprocessing._deskew_evidence(
        mask,
        cv2.findNonZero(mask),
        cv2.getRotationMatrix2D((500, 150), 3, 1),
    )
    assert evidence["accepted"] is False
    assert evidence["reason"] == "alignment-not-improved"
    assert evidence["foreground_within_canvas"] is True
    assert evidence["candidate_score"] < evidence["baseline_score"]


def test_foreground_crossing_canvas_is_rejected_before_warp(monkeypatch):
    mask = np.zeros((100, 200), np.uint8)
    mask[20:80, 0:20] = 255
    matrix = np.array([[1.0, 0.0, -2.0], [0.0, 1.0, 0.0]])

    def unexpected_warp(*args, **kwargs):
        raise AssertionError("Do not rasterize a candidate known to clip foreground")

    monkeypatch.setattr(preprocessing.cv2, "warpAffine", unexpected_warp)
    evidence = preprocessing._deskew_evidence(mask, cv2.findNonZero(mask), matrix)
    assert evidence["accepted"] is False
    assert evidence["reason"] == "foreground-outside-canvas"
    assert evidence["foreground_within_canvas"] is False
    assert evidence["candidate_score"] is None


def test_equal_alignment_does_not_justify_interpolation():
    mask = _mask(_rows())
    evidence = preprocessing._deskew_evidence(
        mask,
        cv2.findNonZero(mask),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    assert evidence["accepted"] is False
    assert evidence["baseline_score"] == evidence["candidate_score"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_angle_never_reaches_warp(monkeypatch, value):
    image = _rows()
    monkeypatch.setattr(preprocessing.cv2, "minAreaRect", lambda _: ((0, 0), (1, 1), value))
    monkeypatch.setattr(
        preprocessing.cv2,
        "getRotationMatrix2D",
        lambda *args: pytest.fail("Non-finite proposals must be rejected"),
    )
    np.testing.assert_array_equal(preprocessing._deskew(image), image)


@pytest.mark.parametrize("auto_orient", [False, True])
def test_guard_preserves_contrast_cleanup_and_calls_engine_once(tmp_path, auto_orient):
    from lao_document_ocr.models import BoundingBox
    from lao_document_ocr.ocr.base import OcrEngine, RecognizedLine
    from lao_document_ocr.pipeline import process_document

    image = _rows()
    cv2.circle(image, (100, 20), 12, 0, -1)
    pil = Image.fromarray(image)
    expected = ImageEnhance.Contrast(ImageOps.autocontrast(pil, cutoff=0.5)).enhance(1.15)
    seen = []

    class Engine(OcrEngine):
        def is_available(self):
            return True

        def recognize(self, received):
            seen.append(received.copy())
            np.testing.assert_array_equal(np.asarray(received), np.asarray(expected))
            return [
                RecognizedLine(
                    text="Project authored fixture text",
                    bbox=BoundingBox(x=150, y=85, width=150, height=20),
                    confidence=0.99,
                    block_id=1,
                    paragraph_id=1,
                    line_id=1,
                )
            ]

    path = tmp_path / "own-rows-with-corner-graphic.png"
    pil.save(path)
    result = process_document(
        path,
        engine=Engine(),
        auto_orient_right_angles=auto_orient,
    )
    assert len(seen) == 1
    assert (result.pages[0].width, result.pages[0].height) == pil.size


def test_alignment_score_is_finite_for_empty_masks():
    assert preprocessing._horizontal_alignment_score(np.zeros((10, 20), np.uint8)) == 0.0


def test_alignment_score_counts_foreground_not_intensity():
    mask = _mask(_rows())
    assert preprocessing._horizontal_alignment_score(mask) == (
        preprocessing._horizontal_alignment_score(mask // 255)
    )
