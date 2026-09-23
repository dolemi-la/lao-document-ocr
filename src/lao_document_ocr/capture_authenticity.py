from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class CaptureAuthenticity:
    aspect_ratio_delta: float
    normalized_mae: float
    dhash_distance: int
    foreground_fraction: float
    likely_digital_copy: bool
    has_visible_content: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _load_gray(path: str | Path) -> Image.Image:
    try:
        with Image.open(path) as source:
            return ImageOps.exif_transpose(source).convert("L")
    except Exception as exc:
        raise ValueError(f"Could not open capture image: {path}") from exc


def _normalized_array(image: Image.Image, size: int = 64) -> np.ndarray:
    resized = image.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float32)


def _dhash(image: Image.Image) -> int:
    resized = image.resize((9, 8), Image.Resampling.LANCZOS)
    values = np.asarray(resized, dtype=np.uint8)
    comparisons = values[:, 1:] > values[:, :-1]
    result = 0
    for bit in comparisons.flatten():
        result = (result << 1) | int(bool(bit))
    return result


def compare_capture_to_digital_source(
    digital_page: str | Path,
    capture_image: str | Path,
    *,
    max_aspect_ratio_delta: float = 0.005,
    max_normalized_mae: float = 3.0,
    max_dhash_distance: int = 2,
    min_foreground_fraction: float = 0.003,
) -> CaptureAuthenticity:
    if max_aspect_ratio_delta < 0:
        raise ValueError("max_aspect_ratio_delta must be non-negative")
    if max_normalized_mae < 0:
        raise ValueError("max_normalized_mae must be non-negative")
    if max_dhash_distance < 0:
        raise ValueError("max_dhash_distance must be non-negative")
    if not 0 <= min_foreground_fraction < 1:
        raise ValueError("min_foreground_fraction must be in [0, 1)")

    digital = _load_gray(digital_page)
    capture = _load_gray(capture_image)
    if digital.width < 1 or digital.height < 1:
        raise ValueError("Digital source has invalid dimensions")
    if capture.width < 1 or capture.height < 1:
        raise ValueError("Capture image has invalid dimensions")

    digital_ratio = digital.width / digital.height
    capture_ratio = capture.width / capture.height
    aspect_ratio_delta = abs(capture_ratio - digital_ratio) / digital_ratio

    digital_array = _normalized_array(digital)
    capture_array = _normalized_array(capture)
    normalized_mae = float(
        np.mean(np.abs(digital_array - capture_array))
    )

    dhash_distance = (
        _dhash(digital) ^ _dhash(capture)
    ).bit_count()

    capture_full = np.asarray(capture, dtype=np.uint8)
    foreground_fraction = float(np.mean(capture_full < 245))
    has_visible_content = foreground_fraction >= min_foreground_fraction

    likely_digital_copy = (
        aspect_ratio_delta <= max_aspect_ratio_delta
        and normalized_mae <= max_normalized_mae
        and dhash_distance <= max_dhash_distance
    )

    return CaptureAuthenticity(
        aspect_ratio_delta=aspect_ratio_delta,
        normalized_mae=normalized_mae,
        dhash_distance=dhash_distance,
        foreground_fraction=foreground_fraction,
        likely_digital_copy=likely_digital_copy,
        has_visible_content=has_visible_content,
    )
