from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


class AugmentationProfile(StrEnum):
    DEFAULT = "default"
    CLEAN_SCAN = "clean-scan"
    NOISY_SCAN = "noisy-scan"
    PHONE_PHOTO = "phone-photo"
    BALANCED = "balanced"


@dataclass(frozen=True)
class AugmentationConfig:
    max_rotation_degrees: float = 1.5
    noise_std: float = 4.0
    blur_radius: float = 0.35
    brightness_jitter: float = 0.08
    contrast_jitter: float = 0.0
    perspective_jitter: float = 0.0
    shadow_strength: float = 0.0
    min_scale: float = 1.0
    jpeg_quality_min: int = 100
    jpeg_quality_max: int = 100

    def __post_init__(self) -> None:
        if self.max_rotation_degrees < 0:
            raise ValueError("max_rotation_degrees must be non-negative")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if self.blur_radius < 0:
            raise ValueError("blur_radius must be non-negative")
        if not 0 <= self.brightness_jitter < 1:
            raise ValueError("brightness_jitter must be in [0, 1)")
        if not 0 <= self.contrast_jitter < 1:
            raise ValueError("contrast_jitter must be in [0, 1)")
        if not 0 <= self.perspective_jitter <= 0.15:
            raise ValueError("perspective_jitter must be in [0, 0.15]")
        if not 0 <= self.shadow_strength <= 0.5:
            raise ValueError("shadow_strength must be in [0, 0.5]")
        if not 0.25 <= self.min_scale <= 1:
            raise ValueError("min_scale must be in [0.25, 1]")
        if not 1 <= self.jpeg_quality_min <= 100:
            raise ValueError("jpeg_quality_min must be in [1, 100]")
        if not 1 <= self.jpeg_quality_max <= 100:
            raise ValueError("jpeg_quality_max must be in [1, 100]")
        if self.jpeg_quality_min > self.jpeg_quality_max:
            raise ValueError("jpeg_quality_min must be <= jpeg_quality_max")


@dataclass(frozen=True)
class SyntheticSample:
    id: str
    image: str
    text: str
    font: str
    font_size: int
    seed: int
    augmentation_profile: str
    augmentation: dict[str, float | int]
    sha256: str


def augmentation_config_for_profile(
    profile: AugmentationProfile | str,
) -> AugmentationConfig:
    profile = AugmentationProfile(profile)
    if profile == AugmentationProfile.DEFAULT:
        return AugmentationConfig()
    if profile == AugmentationProfile.CLEAN_SCAN:
        return AugmentationConfig(
            max_rotation_degrees=0.45,
            noise_std=1.5,
            blur_radius=0.15,
            brightness_jitter=0.03,
            contrast_jitter=0.04,
            min_scale=0.9,
            jpeg_quality_min=92,
            jpeg_quality_max=100,
        )
    if profile == AugmentationProfile.NOISY_SCAN:
        return AugmentationConfig(
            max_rotation_degrees=2.0,
            noise_std=8.0,
            blur_radius=0.75,
            brightness_jitter=0.12,
            contrast_jitter=0.16,
            shadow_strength=0.06,
            min_scale=0.6,
            jpeg_quality_min=55,
            jpeg_quality_max=90,
        )
    if profile == AugmentationProfile.PHONE_PHOTO:
        return AugmentationConfig(
            max_rotation_degrees=3.0,
            noise_std=5.0,
            blur_radius=0.6,
            brightness_jitter=0.15,
            contrast_jitter=0.12,
            perspective_jitter=0.045,
            shadow_strength=0.2,
            min_scale=0.72,
            jpeg_quality_min=65,
            jpeg_quality_max=95,
        )
    raise ValueError("balanced is a mixture profile and has no single config")


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


def _apply_perspective(
    array: np.ndarray,
    rng: np.random.Generator,
    jitter: float,
) -> tuple[np.ndarray, float]:
    if jitter <= 0:
        return array, 0.0

    height, width = array.shape
    max_x = max(1.0, width * jitter)
    max_y = max(1.0, height * min(jitter * 1.5, 0.12))
    source = np.float32(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ]
    )
    offsets = np.column_stack(
        (
            rng.uniform(-max_x, max_x, size=4),
            rng.uniform(-max_y, max_y, size=4),
        )
    ).astype(np.float32)
    target = source + offsets
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(
        array,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return warped, float(np.max(np.abs(offsets[:, 0])) / max(1, width))


def _apply_shadow(
    array: np.ndarray,
    rng: np.random.Generator,
    strength: float,
) -> tuple[np.ndarray, float]:
    if strength <= 0:
        return array, 0.0

    actual = float(rng.uniform(0, strength))
    height, width = array.shape
    horizontal = bool(rng.integers(0, 2))
    reverse = bool(rng.integers(0, 2))
    length = width if horizontal else height
    gradient = np.linspace(0.0, actual, length, dtype=np.float32)
    if reverse:
        gradient = gradient[::-1]
    if horizontal:
        shade = np.tile(gradient, (height, 1))
    else:
        shade = np.tile(gradient[:, None], (1, width))
    output = array * (1.0 - shade)
    return np.clip(output, 0, 255), actual


def _apply_resolution_loss(
    image: Image.Image,
    rng: np.random.Generator,
    min_scale: float,
) -> tuple[Image.Image, float]:
    if min_scale >= 0.999:
        return image, 1.0

    scale = float(rng.uniform(min_scale, 1.0))
    if scale >= 0.995:
        return image, scale
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    reduced = image.resize((width, height), Image.Resampling.BILINEAR)
    restored = reduced.resize(image.size, Image.Resampling.BICUBIC)
    return restored, scale


def _apply_jpeg(
    image: Image.Image,
    rng: np.random.Generator,
    quality_min: int,
    quality_max: int,
) -> tuple[Image.Image, int]:
    quality = int(rng.integers(quality_min, quality_max + 1))
    if quality >= 100:
        return image, quality
    buffer = io.BytesIO()
    image.convert("L").save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=False,
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("L"), quality


def augment_scan(
    image: Image.Image,
    *,
    seed: int,
    config: AugmentationConfig | None = None,
) -> tuple[Image.Image, dict[str, float | int]]:
    config = config or AugmentationConfig()
    rng = np.random.default_rng(seed)

    angle = float(rng.uniform(-config.max_rotation_degrees, config.max_rotation_degrees))
    brightness = float(rng.uniform(1 - config.brightness_jitter, 1 + config.brightness_jitter))
    contrast = float(rng.uniform(1 - config.contrast_jitter, 1 + config.contrast_jitter))
    blur = float(rng.uniform(0, config.blur_radius))
    noise_std = float(rng.uniform(0, config.noise_std))

    array = np.asarray(image.convert("L"), dtype=np.float32)
    array = np.clip((array - 127.5) * contrast + 127.5, 0, 255)
    array = np.clip(array * brightness, 0, 255)
    array, shadow = _apply_shadow(array, rng, config.shadow_strength)

    if noise_std > 0:
        array += rng.normal(0, noise_std, size=array.shape)
        array = np.clip(array, 0, 255)

    if abs(angle) > 0.01:
        height, width = array.shape
        center = (width / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        array = cv2.warpAffine(
            array,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

    array, perspective = _apply_perspective(
        array,
        rng,
        config.perspective_jitter,
    )
    output = Image.fromarray(array.astype(np.uint8))

    if blur > 0.01:
        output = output.filter(ImageFilter.GaussianBlur(radius=blur))

    output, scale = _apply_resolution_loss(output, rng, config.min_scale)
    output, jpeg_quality = _apply_jpeg(
        output,
        rng,
        config.jpeg_quality_min,
        config.jpeg_quality_max,
    )

    metadata: dict[str, float | int] = {
        "rotation_degrees": angle,
        "brightness": brightness,
        "contrast": contrast,
        "blur_radius": blur,
        "noise_std": noise_std,
        "perspective_ratio": perspective,
        "shadow_strength": shadow,
        "resolution_scale": scale,
        "jpeg_quality": jpeg_quality,
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


_BALANCED_PROFILES = (
    AugmentationProfile.CLEAN_SCAN,
    AugmentationProfile.NOISY_SCAN,
    AugmentationProfile.PHONE_PHOTO,
)


def _resolve_profile(
    requested: AugmentationProfile,
    sample_index: int,
) -> AugmentationProfile:
    if requested != AugmentationProfile.BALANCED:
        return requested
    return _BALANCED_PROFILES[sample_index % len(_BALANCED_PROFILES)]


def _font_index(
    *,
    requested_profile: AugmentationProfile,
    sample_index: int,
    line_index: int,
    variant: int,
    font_count: int,
) -> int:
    if requested_profile != AugmentationProfile.BALANCED:
        return (line_index + variant) % font_count

    profile_count = len(_BALANCED_PROFILES)
    cycle_index, profile_index = divmod(sample_index, profile_count)
    return (cycle_index + profile_index) % font_count


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
    start_line: int = 0,
    augmentation: AugmentationConfig | None = None,
    augmentation_profile: AugmentationProfile | str = AugmentationProfile.DEFAULT,
) -> Path:
    if not corpus_lines:
        raise ValueError("corpus is empty")
    if not font_paths:
        raise ValueError("at least one font is required")
    if variants_per_line < 1:
        raise ValueError("variants_per_line must be at least 1")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be at least 1")
    if start_line < 0 or start_line >= len(corpus_lines):
        raise ValueError("start_line must be within the corpus")

    requested_profile = AugmentationProfile(augmentation_profile)
    if augmentation is not None and requested_profile != AugmentationProfile.DEFAULT:
        raise ValueError(
            "augmentation and augmentation_profile cannot both override defaults"
        )

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
    prior_nonblank_lines = sum(1 for text in corpus_lines[:start_line] if text.strip())
    sample_index = prior_nonblank_lines * variants_per_line
    emitted_samples = 0

    for line_index, text in enumerate(corpus_lines[start_line:], start=start_line):
        if not text.strip():
            continue
        for variant in range(variants_per_line):
            if max_samples is not None and emitted_samples >= max_samples:
                manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")
                return manifest

            font = fonts[
                _font_index(
                    requested_profile=requested_profile,
                    sample_index=sample_index,
                    line_index=line_index,
                    variant=variant,
                    font_count=len(fonts),
                )
            ]
            font_size = sizes[(line_index + variant) % len(sizes)]
            sample_seed = seed + sample_index
            sample_id = f"line-{sample_index:08d}"
            profile = _resolve_profile(requested_profile, sample_index)
            config = (
                augmentation
                if augmentation is not None
                else augmentation_config_for_profile(profile)
            )

            image = render_text_line(text, font, font_size=font_size)
            image, aug_meta = augment_scan(
                image,
                seed=sample_seed,
                config=config,
            )

            image_path = images_dir / f"{sample_id}.png"
            image.convert("L").save(image_path, format="PNG", optimize=True)

            sample = SyntheticSample(
                id=sample_id,
                image=str(image_path.relative_to(output)),
                text=text,
                font=font.name,
                font_size=font_size,
                seed=sample_seed,
                augmentation_profile=profile.value,
                augmentation=aug_meta,
                sha256=_sha256(image_path),
            )
            entries.append(json.dumps(asdict(sample), ensure_ascii=False))
            sample_index += 1
            emitted_samples += 1

    manifest.write_text(
        "\n".join(entries) + ("\n" if entries else ""),
        encoding="utf-8",
    )
    return manifest


def load_corpus(path: str | Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
