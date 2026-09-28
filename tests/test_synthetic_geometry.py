from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from lao_document_ocr.synthetic import AugmentationConfig, _apply_perspective, augment_scan


def _corner_markers(width: int = 1200, height: int = 80) -> np.ndarray:
    image = np.full((height, width), 255, dtype=np.float32)
    for x in (2, width - 12):
        for y in (2, height - 12):
            image[y : y + 10, x : x + 10] = 0
    return image


def _assert_four_complete_markers(image: np.ndarray) -> None:
    binary = (image < 128).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    assert count - 1 == 4
    # Content must not touch the expanded boundary. This catches partial clipping
    # even when all four components survive.
    for x, y, width, height, area in stats[1:]:
        assert x > 0 and y > 0
        assert x + width < image.shape[1]
        assert y + height < image.shape[0]
        assert area > 20


@pytest.mark.parametrize("seed", [2, 8, 42])
def test_rotation_preserves_edge_content_and_records_geometry(seed) -> None:
    original = _corner_markers()
    config = AugmentationConfig(
        max_rotation_degrees=3,
        noise_std=0,
        blur_radius=0,
        brightness_jitter=0,
    )
    result, metadata = augment_scan(
        Image.fromarray(original.astype(np.uint8)), seed=seed, config=config
    )
    grayscale = np.asarray(result.convert("L"))
    _assert_four_complete_markers(grayscale)
    # Rotation without scaling should preserve the integrated ink mass apart
    # from interpolation/quantization, not silently remove entire edge glyphs.
    original_ink = float((255 - original).sum())
    result_ink = float((255 - grayscale.astype(np.float32)).sum())
    assert result_ink / original_ink == pytest.approx(1.0, abs=0.03)
    assert metadata["geometry_version"] == 2
    assert metadata["source_width"] == original.shape[1]
    assert metadata["source_height"] == original.shape[0]
    assert metadata["output_width"] == result.width
    assert metadata["output_height"] == result.height


@pytest.mark.parametrize("seed", [0, 2, 42])
def test_perspective_preserves_outward_transformed_edge_markers(seed) -> None:
    original = _corner_markers(width=400, height=120)
    output, ratio = _apply_perspective(original, np.random.default_rng(seed), 0.1)
    assert 0 < ratio <= 0.1
    _assert_four_complete_markers(output)


@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (4, 1)])
def test_perspective_does_not_warp_degenerate_images(shape) -> None:
    original = np.full(shape, 127, dtype=np.float32)
    output, ratio = _apply_perspective(original, np.random.default_rng(42), 0.1)
    np.testing.assert_array_equal(output, original)
    assert ratio == 0.0


def test_identity_augmentation_does_not_change_dimensions_or_pixels() -> None:
    original = _corner_markers().astype(np.uint8)
    result, metadata = augment_scan(
        Image.fromarray(original),
        seed=42,
        config=AugmentationConfig(
            max_rotation_degrees=0, noise_std=0, blur_radius=0, brightness_jitter=0
        ),
    )
    np.testing.assert_array_equal(np.asarray(result.convert("L")), original)
    assert result.size == (original.shape[1], original.shape[0])
    assert metadata["geometry_version"] == 2


def test_interpolation_overshoot_is_clamped_not_wrapped(monkeypatch) -> None:
    import lao_document_ocr.synthetic as module

    def overshoot(array, rng, jitter):
        return np.array([[-20.0, 0.0, 255.0, 275.0]], dtype=np.float32), 0.01

    monkeypatch.setattr(module, "_apply_perspective", overshoot)
    result, _ = augment_scan(
        Image.new("L", (4, 1), 255),
        seed=42,
        config=AugmentationConfig(
            max_rotation_degrees=0, noise_std=0, blur_radius=0, brightness_jitter=0
        ),
    )
    np.testing.assert_array_equal(np.asarray(result.convert("L")), [[0, 0, 255, 255]])


def test_combined_warps_keep_edge_markers() -> None:
    original = _corner_markers(width=900, height=150).astype(np.uint8)
    result, metadata = augment_scan(
        Image.fromarray(original),
        seed=42,
        config=AugmentationConfig(
            max_rotation_degrees=3,
            perspective_jitter=0.06,
            noise_std=0,
            blur_radius=0,
            brightness_jitter=0,
        ),
    )
    _assert_four_complete_markers(np.asarray(result.convert("L")))
    assert metadata["output_width"] == result.width
    assert metadata["output_height"] == result.height


def test_geometry_does_not_invent_ink_in_uniform_white() -> None:
    result, _ = augment_scan(
        Image.new("L", (1200, 80), 255),
        seed=42,
        config=AugmentationConfig(
            max_rotation_degrees=3,
            perspective_jitter=0.06,
            noise_std=0,
            blur_radius=0,
            brightness_jitter=0,
        ),
    )
    assert result.convert("L").getextrema() == (255, 255)


def test_projective_horizon_crossing_is_rejected() -> None:
    from lao_document_ocr.synthetic import _warp_full_canvas

    transform = np.array([[1, 0, 0], [0, 1, 0], [1, 0, -50]], dtype=np.float64)
    with pytest.raises(ValueError, match="projective singularity"):
        _warp_full_canvas(np.full((80, 100), 255, dtype=np.float32), transform)
