"""Small-angle page cleanup must undo tilt, not compound it."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

import lao_document_ocr.preprocessing as preprocessing


def _printed_rows() -> np.ndarray:
    image = np.full((300, 1000), 255, dtype=np.uint8)
    for y in (85, 145, 205):
        for x in range(150, 850, 30):
            cv2.rectangle(image, (x, y - 8), (x + 15, y + 8), 0, -1)
    return image


def _residual_tilt(image: np.ndarray) -> float:
    points = cv2.findNonZero((image < 128).astype(np.uint8))
    assert points is not None
    return (float(cv2.minAreaRect(points)[-1]) + 45.0) % 90.0 - 45.0


@pytest.mark.parametrize("tilt", [-10.0, -6.0, -3.0, -1.0, 1.0, 3.0, 6.0, 10.0])
def test_deskew_straightens_both_rotation_directions(tilt):
    original = _printed_rows()
    height, width = original.shape
    tilted = cv2.warpAffine(
        original,
        cv2.getRotationMatrix2D((width / 2, height / 2), tilt, 1.0),
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderValue=255,
    )
    assert abs(_residual_tilt(tilted)) == pytest.approx(abs(tilt), abs=0.1)

    corrected = preprocessing._deskew(tilted)

    assert abs(_residual_tilt(corrected)) < 0.15
    assert corrected.shape == original.shape
    assert corrected.dtype == np.uint8


@pytest.mark.parametrize("tilt", [-3.0, 3.0])
def test_public_preprocess_straightens_tilt_without_resizing_page(tilt):
    image = _printed_rows()
    height, width = image.shape
    tilted = cv2.warpAffine(
        image,
        cv2.getRotationMatrix2D((width / 2, height / 2), tilt, 1.0),
        (width, height),
        borderValue=255,
    )
    result = preprocessing.preprocess_image(Image.fromarray(tilted).convert("RGB"))
    assert result.mode == "L"
    assert result.size == (width, height)
    assert abs(_residual_tilt(np.asarray(result))) < 0.15


@pytest.mark.parametrize("kind", ["upright", "blank", "sparse"])
def test_deskew_leaves_upright_blank_and_sparse_images_unchanged(kind):
    image = _printed_rows() if kind == "upright" else np.full((100, 200), 255, np.uint8)
    if kind == "sparse":
        image[30:32, 40:50] = 0
    np.testing.assert_array_equal(preprocessing._deskew(image), image)


@pytest.mark.parametrize(
    "reported,expected",
    [
        (-87.0, 3.0),
        (3.0, 3.0),
        (93.0, 3.0),
        (-3.0, -3.0),
        (87.0, -3.0),
        (-93.0, -3.0),
    ],
)
def test_equivalent_rectangle_angle_conventions_use_same_correction(
    monkeypatch,
    reported,
    expected,
):
    image = _printed_rows()
    recorded = {}
    # Isolate the direction convention from the separate image-evidence gate.
    monkeypatch.setattr(preprocessing, "_deskew_evidence", lambda *args: {"accepted": True})
    monkeypatch.setattr(preprocessing.cv2, "minAreaRect", lambda _: ((0, 0), (100, 10), reported))

    def warp(array, matrix, size, **kwargs):
        recorded["matrix"] = matrix
        recorded["size"] = size
        return array

    monkeypatch.setattr(preprocessing.cv2, "warpAffine", warp)
    preprocessing._deskew(image)
    np.testing.assert_allclose(
        recorded["matrix"],
        cv2.getRotationMatrix2D((500, 150), expected, 1.0),
    )
    assert recorded["size"] == (1000, 300)


@pytest.mark.parametrize("reported", [-90.0, 0.0, 90.0, -0.1, 0.1, -13.0, 13.0, 45.0])
def test_deskew_keeps_existing_small_angle_guards(monkeypatch, reported):
    image = _printed_rows()
    monkeypatch.setattr(preprocessing.cv2, "minAreaRect", lambda _: ((0, 0), (100, 10), reported))

    def unexpected_warp(*args, **kwargs):
        raise AssertionError("Upright or out-of-range estimates must not rotate")

    monkeypatch.setattr(preprocessing.cv2, "warpAffine", unexpected_warp)
    np.testing.assert_array_equal(preprocessing._deskew(image), image)


@pytest.mark.parametrize("tilt", [-3.0, 3.0])
def test_document_pipeline_passes_corrected_pixels_to_engine(tmp_path, tilt):
    from lao_document_ocr.ocr.base import OcrEngine
    from lao_document_ocr.pipeline import process_document

    class InspectingEngine(OcrEngine):
        def is_available(self):
            return True

        def recognize(self, image):
            assert image.size == (1000, 300)
            assert abs(_residual_tilt(np.asarray(image.convert("L")))) < 0.15
            return []

    image = _printed_rows()
    tilted = cv2.warpAffine(
        image,
        cv2.getRotationMatrix2D((500, 150), tilt, 1.0),
        (1000, 300),
        borderValue=255,
    )
    source = tmp_path / "tilted-page.png"
    Image.fromarray(tilted).save(source)
    document = process_document(source, engine=InspectingEngine())
    assert len(document.pages) == 1
    assert document.pages[0].width == 1000
    assert document.pages[0].height == 300
