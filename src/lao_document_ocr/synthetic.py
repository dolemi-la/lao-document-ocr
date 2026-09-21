from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


@dataclass(frozen=True)
class AugmentationConfig:
    max_rotation_degrees: float = 1.5
    noise_std: float = 4.0
    blur_radius: float = 0.35
    brightness_jitter: float = 0.08


@dataclass(frozen=True)
class SyntheticSample:
    id: str
    image: str
    text: str
    font: str
    font_size: int
    seed: int
    augmentation: dict[str, float]
    sha256: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_text_line(
    text: str,
    font_path: str | Path,
    *,
    font_size: int = 48,
    padding_x: int = 32,
    padding_y: int = 20,
) -> Image.Image:
    font = ImageFont.truetype(str(font_path), size=font_size)
    probe = Image.new("L", (16, 16), 255)
    draw = ImageDraw.Draw(probe)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    width = max(1, right - left)
    height = max(1, bottom - top)

    canvas = Image.new("L", (width + padding_x * 2, height + padding_y * 2), 255)
    draw = ImageDraw.Draw(canvas)
    draw.text((padding_x - left, padding_y - top), text, font=font, fill=0)
    return canvas


def augment_scan(
    image: Image.Image,
    *,
    seed: int,
    config: AugmentationConfig | None = None,
) -> tuple[Image.Image, dict[str, float]]:
    config = config or AugmentationConfig()
    rng = np.random.default_rng(seed)

    angle = float(rng.uniform(-config.max_rotation_degrees, config.max_rotation_degrees))
    brightness = float(rng.uniform(1 - config.brightness_jitter, 1 + config.brightness_jitter))
    blur = float(rng.uniform(0, config.blur_radius))
    noise_std = float(rng.uniform(0, config.noise_std))

    array = np.asarray(image.convert("L"), dtype=np.float32)
    array = np.clip(array * brightness, 0, 255)

    if noise_std > 0:
        array += rng.normal(0, noise_std, size=array.shape)
        array = np.clip(array, 0, 255)

    output = Image.fromarray(array.astype(np.uint8))

    if blur > 0.01:
        output = output.filter(ImageFilter.GaussianBlur(radius=blur))

    if abs(angle) > 0.01:
        arr = np.asarray(output)
        height, width = arr.shape
        center = (width / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            arr,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        output = Image.fromarray(rotated)

    metadata = {
        "rotation_degrees": angle,
        "brightness": brightness,
        "blur_radius": blur,
        "noise_std": noise_std,
    }
    return output.convert("RGB"), metadata


def _font_sizes(min_size: int, max_size: int) -> list[int]:
    if min_size < 8:
        raise ValueError("min font size must be at least 8")
    if max_size < min_size:
        raise ValueError("max font size must be >= min font size")
    if min_size == max_size:
        return [min_size]
    step = max(1, math.ceil((max_size - min_size) / 4))
    sizes = list(range(min_size, max_size + 1, step))
    if sizes[-1] != max_size:
        sizes.append(max_size)
    return sizes


def generate_synthetic_lines(
    corpus_lines: list[str],
    output_dir: str | Path,
    font_paths: list[str | Path],
    *,
    variants_per_line: int = 1,
    seed: int = 20260921,
    min_font_size: int = 40,
    max_font_size: int = 56,
    max_samples: int | None = None,
    augmentation: AugmentationConfig | None = None,
) -> Path:
    if not corpus_lines:
        raise ValueError("corpus is empty")
    if not font_paths:
        raise ValueError("at least one font is required")
    if variants_per_line < 1:
        raise ValueError("variants_per_line must be at least 1")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be at least 1")

    fonts = [Path(path) for path in font_paths]
    missing = [str(path) for path in fonts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing font files: {', '.join(missing)}")

    sizes = _font_sizes(min_font_size, max_font_size)
    output = Path(output_dir)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.jsonl"

    entries: list[str] = []
    sample_index = 0

    for line_index, text in enumerate(corpus_lines):
        if not text.strip():
            continue
        for variant in range(variants_per_line):
            if max_samples is not None and sample_index >= max_samples:
                manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")
                return manifest

            font = fonts[(line_index + variant) % len(fonts)]
            font_size = sizes[(line_index + variant) % len(sizes)]
            sample_seed = seed + sample_index
            sample_id = f"line-{sample_index:08d}"

            image = render_text_line(text, font, font_size=font_size)
            image, aug_meta = augment_scan(image, seed=sample_seed, config=augmentation)

            image_path = images_dir / f"{sample_id}.png"
            image.save(image_path, format="PNG", optimize=True)

            sample = SyntheticSample(
                id=sample_id,
                image=str(image_path.relative_to(output)),
                text=text,
                font=font.name,
                font_size=font_size,
                seed=sample_seed,
                augmentation=aug_meta,
                sha256=_sha256(image_path),
            )
            entries.append(json.dumps(asdict(sample), ensure_ascii=False))
            sample_index += 1

    manifest.write_text("\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")
    return manifest


def load_corpus(path: str | Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
