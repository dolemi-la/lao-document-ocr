from __future__ import annotations

import hashlib
import json
import shutil
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


def test_generate_synthetic_lines_chunk_matches_full_run(tmp_path) -> None:
    corpus = ["first", "", "second", "third", "fourth"]
    source_font = _font_path()
    fonts = []
    for name in ("chunk-font-a.ttf", "chunk-font-b.ttf", "chunk-font-c.ttf"):
        path = tmp_path / name
        shutil.copyfile(source_font, path)
        fonts.append(path)

    full = generate_synthetic_lines(
        corpus,
        tmp_path / "full",
        fonts,
        variants_per_line=1,
        seed=77,
        min_font_size=24,
        max_font_size=24,
        augmentation_profile="balanced",
    )
    chunk = generate_synthetic_lines(
        corpus,
        tmp_path / "chunk",
        fonts,
        variants_per_line=1,
        seed=77,
        min_font_size=24,
        max_font_size=24,
        max_samples=2,
        start_line=2,
        augmentation_profile="balanced",
    )

    full_entries = [json.loads(line) for line in full.read_text().splitlines()]
    chunk_entries = [json.loads(line) for line in chunk.read_text().splitlines()]

    assert [entry["id"] for entry in chunk_entries] == [
        full_entries[1]["id"],
        full_entries[2]["id"],
    ]
    assert [entry["seed"] for entry in chunk_entries] == [
        full_entries[1]["seed"],
        full_entries[2]["seed"],
    ]
    assert [entry["augmentation_profile"] for entry in chunk_entries] == [
        full_entries[1]["augmentation_profile"],
        full_entries[2]["augmentation_profile"],
    ]
    assert [entry["font"] for entry in chunk_entries] == [
        full_entries[1]["font"],
        full_entries[2]["font"],
    ]
    assert [entry["sha256"] for entry in chunk_entries] == [
        full_entries[1]["sha256"],
        full_entries[2]["sha256"],
    ]


def test_generate_synthetic_lines_rejects_invalid_start_line(tmp_path) -> None:
    with pytest.raises(ValueError, match="start_line"):
        generate_synthetic_lines(
            ["OCR"],
            tmp_path / "negative",
            [_font_path()],
            start_line=-1,
        )
    with pytest.raises(ValueError, match="start_line"):
        generate_synthetic_lines(
            ["OCR"],
            tmp_path / "past-end",
            [_font_path()],
            start_line=1,
        )


def test_generate_synthetic_lines_store_grayscale_pngs(tmp_path) -> None:
    manifest = generate_synthetic_lines(
        ["OCR grayscale"],
        tmp_path / "grayscale",
        [_font_path()],
        min_font_size=24,
        max_font_size=24,
    )
    entry = json.loads(manifest.read_text(encoding="utf-8").strip())

    from PIL import Image

    with Image.open(manifest.parent / entry["image"]) as image:
        assert image.mode == "L"


def test_generate_synthetic_lines_rejects_missing_font(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        generate_synthetic_lines(
            ["OCR"],
            tmp_path / "dataset",
            [tmp_path / "missing.ttf"],
        )


def test_phone_profile_is_deterministic_and_records_realistic_artifacts() -> None:
    from lao_document_ocr.synthetic import (
        AugmentationProfile,
        augmentation_config_for_profile,
    )

    image = render_text_line("Phone OCR", _font_path(), font_size=32)
    config = augmentation_config_for_profile(AugmentationProfile.PHONE_PHOTO)

    first, first_meta = augment_scan(image, seed=123, config=config)
    second, second_meta = augment_scan(image, seed=123, config=config)

    assert first_meta == second_meta
    assert hashlib.sha256(first.tobytes()).hexdigest() == hashlib.sha256(
        second.tobytes()
    ).hexdigest()
    assert 0.72 <= first_meta["resolution_scale"] <= 1.0
    assert 65 <= first_meta["jpeg_quality"] <= 95
    assert 0 <= first_meta["shadow_strength"] <= 0.2
    assert 0 <= first_meta["perspective_ratio"] <= 0.045


def test_profile_presets_have_distinct_severity() -> None:
    from lao_document_ocr.synthetic import (
        AugmentationProfile,
        augmentation_config_for_profile,
    )

    clean = augmentation_config_for_profile(AugmentationProfile.CLEAN_SCAN)
    noisy = augmentation_config_for_profile(AugmentationProfile.NOISY_SCAN)
    phone = augmentation_config_for_profile(AugmentationProfile.PHONE_PHOTO)

    assert clean.noise_std < noisy.noise_std
    assert clean.jpeg_quality_min > noisy.jpeg_quality_min
    assert phone.perspective_jitter > 0
    assert phone.shadow_strength > noisy.shadow_strength


def test_balanced_profile_crosses_fonts_and_capture_modes(tmp_path) -> None:
    source_font = _font_path()
    fonts = []
    for name in ("font-a.ttf", "font-b.ttf", "font-c.ttf"):
        path = tmp_path / name
        shutil.copyfile(source_font, path)
        fonts.append(path)

    manifest = generate_synthetic_lines(
        [f"line {index}" for index in range(9)],
        tmp_path / "balanced-fonts",
        fonts,
        variants_per_line=1,
        seed=99,
        min_font_size=24,
        max_font_size=24,
        augmentation_profile="balanced",
    )
    entries = [json.loads(line) for line in manifest.read_text().splitlines()]

    pairs = {
        (entry["font"], entry["augmentation_profile"])
        for entry in entries
    }
    assert len(pairs) == 9
    assert pairs == {
        (font.name, profile)
        for font in fonts
        for profile in ("clean-scan", "noisy-scan", "phone-photo")
    }


def test_balanced_profile_cycles_capture_modes(tmp_path) -> None:
    from lao_document_ocr.synthetic import AugmentationProfile

    manifest = generate_synthetic_lines(
        ["A", "B"],
        tmp_path / "balanced",
        [_font_path()],
        variants_per_line=3,
        min_font_size=24,
        max_font_size=24,
        augmentation_profile=AugmentationProfile.BALANCED,
    )
    entries = [json.loads(line) for line in manifest.read_text().splitlines()]

    assert [entry["augmentation_profile"] for entry in entries] == [
        "clean-scan",
        "noisy-scan",
        "phone-photo",
        "clean-scan",
        "noisy-scan",
        "phone-photo",
    ]
    assert all("jpeg_quality" in entry["augmentation"] for entry in entries)


def test_custom_augmentation_and_named_profile_cannot_be_combined(tmp_path) -> None:
    from lao_document_ocr.synthetic import AugmentationProfile

    with pytest.raises(ValueError, match="cannot both override"):
        generate_synthetic_lines(
            ["OCR"],
            tmp_path / "conflict",
            [_font_path()],
            augmentation=AugmentationConfig(),
            augmentation_profile=AugmentationProfile.NOISY_SCAN,
        )
