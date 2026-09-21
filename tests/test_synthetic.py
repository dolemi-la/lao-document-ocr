from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lao_document_ocr.synthetic import (
    AugmentationConfig,
    augment_scan,
    generate_synthetic_lines,
    render_text_line,
)


def _font_path() -> Path:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    pytest.skip("No portable TrueType test font is available")


def test_render_text_line_has_content() -> None:
    image = render_text_line("OCR 123", _font_path(), font_size=32)
    assert image.width > 100
    assert image.height > 30
    assert image.getextrema()[0] < 255


def test_augmentation_is_deterministic() -> None:
    image = render_text_line("OCR", _font_path(), font_size=32)
    config = AugmentationConfig(
        max_rotation_degrees=1,
        noise_std=2,
        blur_radius=0.2,
        brightness_jitter=0.02,
    )
    first, first_meta = augment_scan(image, seed=42, config=config)
    second, second_meta = augment_scan(image, seed=42, config=config)

    assert first_meta == second_meta
    assert hashlib.sha256(first.tobytes()).hexdigest() == hashlib.sha256(
        second.tobytes()
    ).hexdigest()


def test_generate_synthetic_lines_writes_manifest(tmp_path) -> None:
    manifest = generate_synthetic_lines(
        ["Hello OCR", "Document 123"],
        tmp_path / "dataset",
        [_font_path()],
        variants_per_line=2,
        seed=10,
        min_font_size=24,
        max_font_size=28,
    )

    entries = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(entries) == 4
    assert entries[0]["id"] == "line-00000000"
    assert entries[0]["text"] == "Hello OCR"
    assert (manifest.parent / entries[0]["image"]).is_file()


def test_generate_synthetic_lines_rejects_missing_font(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        generate_synthetic_lines(
            ["OCR"],
            tmp_path / "dataset",
            [tmp_path / "missing.ttf"],
        )
