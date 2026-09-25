from __future__ import annotations

import io
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image

from lao_document_ocr.diagram_regions import detect_diagram_regions
from lao_document_ocr.embedded_images import EmbeddedImageAsset, extract_pdf_embedded_images
from lao_document_ocr.header_footer import mark_repeated_headers_footers
from lao_document_ocr.models import BlockType, Document, Page
from lao_document_ocr.ocr.base import OcrEngine, OcrEngineError
from lao_document_ocr.ocr.tesseract import TesseractEngine
from lao_document_ocr.preprocessing import preprocess_image
from lao_document_ocr.raster_regions import detect_raster_regions
from lao_document_ocr.reading_order import (
    DeterministicReadingOrderResolver,
    ReadingOrderResolver,
)
from lao_document_ocr.structure import build_page_blocks

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
SUPPORTED_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | {".pdf"}
DEFAULT_MAX_PAGE_PIXELS = 40_000_000
DEFAULT_AUTO_ORIENT_PROBE_BELOW_CONFIDENCE = 0.65
DEFAULT_AUTO_ORIENT_LAO_DOMINANT_MIN_CONFIDENCE = 0.45
DEFAULT_AUTO_ORIENT_LAO_DOMINANT_MIN_CHARACTERS = 200
DEFAULT_AUTO_ORIENT_LAO_DOMINANT_RATIO = 0.75


class DocumentProcessingError(RuntimeError):
    pass


class DocumentProcessingCancelled(DocumentProcessingError):
    pass


def _raise_if_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise DocumentProcessingCancelled("Document processing was cancelled.")


@dataclass(frozen=True)
class LoadedPage:
    image: Image.Image
    embedded_images: tuple[EmbeddedImageAsset, ...] = ()


def _render_pdf(
    path: Path,
    max_pages: int,
    max_page_pixels: int,
    should_cancel: Callable[[], bool] | None = None,
) -> list[LoadedPage]:
    pages: list[LoadedPage] = []
    try:
        pdf = pymupdf.open(path)
    except Exception as exc:
        raise DocumentProcessingError("Could not open PDF.") from exc

    try:
        if pdf.needs_pass:
            raise DocumentProcessingError("Encrypted PDFs are not supported.")
        if pdf.page_count > max_pages:
            raise DocumentProcessingError(
                f"PDF has {pdf.page_count} pages; maximum is {max_pages}."
            )
        for page in pdf:
            _raise_if_cancelled(should_cancel)
            render_width = math.ceil(float(page.rect.width) * 2)
            render_height = math.ceil(float(page.rect.height) * 2)
            if render_width * render_height > max_page_pixels:
                raise DocumentProcessingError(
                    f"PDF page {page.number + 1} exceeds the rendered pixel limit."
                )
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
            mode = "RGB" if pixmap.n < 4 else "RGBA"
            image = Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)
            embedded = extract_pdf_embedded_images(
                pdf,
                page,
                rendered_width=pixmap.width,
                rendered_height=pixmap.height,
                max_source_pixels=max_page_pixels,
            )
            pages.append(
                LoadedPage(
                    image=image.convert("RGB"),
                    embedded_images=tuple(embedded),
                )
            )
    finally:
        pdf.close()
    return pages


def _load_pages(
    path: Path,
    max_pages: int,
    max_page_pixels: int,
    should_cancel: Callable[[], bool] | None = None,
) -> list[LoadedPage]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _render_pdf(
            path,
            max_pages,
            max_page_pixels,
            should_cancel=should_cancel,
        )
    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        try:
            with Image.open(path) as image:
                if image.width * image.height > max_page_pixels:
                    raise DocumentProcessingError(
                        "Image dimensions exceed the pixel limit."
                    )
                return [LoadedPage(image=image.convert("RGB"))]
        except DocumentProcessingError:
            raise
        except Exception as exc:
            raise DocumentProcessingError("Could not open image.") from exc
    raise DocumentProcessingError(f"Unsupported file type: {suffix or 'unknown'}")


def _rotate_image_clockwise(image: Image.Image, degrees: int) -> Image.Image:
    normalized = degrees % 360
    if normalized == 0:
        return image
    operations = {
        90: Image.Transpose.ROTATE_270,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_90,
    }
    operation = operations.get(normalized)
    if operation is None:
        raise ValueError("Right-angle rotation must be 0, 90, 180, or 270 degrees.")
    return image.transpose(operation)


def _rotate_png_bytes_clockwise(data: bytes, degrees: int) -> bytes:
    normalized = degrees % 360
    if normalized == 0:
        return data
    with Image.open(io.BytesIO(data)) as source:
        rotated = _rotate_image_clockwise(source, normalized)
        output = io.BytesIO()
        rotated.save(output, format="PNG", optimize=True)
        return output.getvalue()


def _rotate_bbox_clockwise(
    bbox,
    *,
    page_width: int,
    page_height: int,
    degrees: int,
):
    from lao_document_ocr.models import BoundingBox

    normalized = degrees % 360
    if normalized == 0:
        return bbox
    if normalized == 90:
        return BoundingBox(
            x=page_height - (bbox.y + bbox.height),
            y=bbox.x,
            width=bbox.height,
            height=bbox.width,
        )
    if normalized == 180:
        return BoundingBox(
            x=page_width - (bbox.x + bbox.width),
            y=page_height - (bbox.y + bbox.height),
            width=bbox.width,
            height=bbox.height,
        )
    if normalized == 270:
        return BoundingBox(
            x=bbox.y,
            y=page_width - (bbox.x + bbox.width),
            width=bbox.height,
            height=bbox.width,
        )
    raise ValueError("Right-angle rotation must be 0, 90, 180, or 270 degrees.")


def _rotate_embedded_assets(
    assets: tuple[EmbeddedImageAsset, ...],
    *,
    page_width: int,
    page_height: int,
    degrees: int,
) -> tuple[EmbeddedImageAsset, ...]:
    normalized = degrees % 360
    if normalized == 0:
        return assets

    rotated_page_width = page_height if normalized in {90, 270} else page_width
    output: list[EmbeddedImageAsset] = []
    for asset in assets:
        bbox = _rotate_bbox_clockwise(
            asset.bbox,
            page_width=page_width,
            page_height=page_height,
            degrees=normalized,
        )
        output.append(
            EmbeddedImageAsset(
                bbox=bbox,
                png_bytes=_rotate_png_bytes_clockwise(asset.png_bytes, normalized),
                width_ratio=bbox.width / max(1, rotated_page_width),
                xref=asset.xref,
            )
        )
    return tuple(sorted(output, key=lambda asset: (asset.bbox.y, asset.bbox.x)))


def _orientation_line_stats(lines) -> tuple[float, int, float, float]:
    total_weight = 0
    weighted_confidence = 0.0
    recognized_characters = 0
    lao_characters = 0
    for line in lines:
        nonspace = [char for char in line.text if not char.isspace()]
        weight = len(nonspace)
        if weight == 0:
            continue
        total_weight += weight
        recognized_characters += weight
        lao_characters += sum(
            1 for char in nonspace if "\u0e80" <= char <= "\u0eff"
        )
        weighted_confidence += float(line.confidence) * weight
    mean_confidence = (
        weighted_confidence / total_weight if total_weight else 0.0
    )
    lao_ratio = (
        lao_characters / recognized_characters
        if recognized_characters
        else 0.0
    )
    score = mean_confidence * math.log1p(recognized_characters)
    return mean_confidence, recognized_characters, score, lao_ratio


def _orientation_candidate_is_acceptable(
    *,
    degrees: int,
    confidence: float,
    characters: int,
    score: float,
    baseline_confidence: float,
    baseline_characters: int,
    baseline_score: float,
) -> bool:
    return (
        degrees != 0
        and score >= max(0.01, baseline_score * 1.15)
        and confidence >= baseline_confidence + 0.08
        and characters >= max(20, int(baseline_characters * 0.5))
    )


def _recognize_with_right_angle_orientation(
    image: Image.Image,
    *,
    engine: OcrEngine,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[Image.Image, list, int, dict[str, object]]:
    baseline_lines = engine.recognize(image)
    (
        baseline_confidence,
        baseline_characters,
        baseline_score,
        baseline_lao_ratio,
    ) = _orientation_line_stats(baseline_lines)

    probe_skip_reason = None
    if baseline_confidence >= DEFAULT_AUTO_ORIENT_PROBE_BELOW_CONFIDENCE:
        probe_skip_reason = "high-confidence"
    elif (
        baseline_confidence >= DEFAULT_AUTO_ORIENT_LAO_DOMINANT_MIN_CONFIDENCE
        and baseline_characters >= DEFAULT_AUTO_ORIENT_LAO_DOMINANT_MIN_CHARACTERS
        and baseline_lao_ratio >= DEFAULT_AUTO_ORIENT_LAO_DOMINANT_RATIO
    ):
        probe_skip_reason = "lao-dominant-baseline"

    base_diagnostics: dict[str, object] = {
        "baseline_confidence": baseline_confidence,
        "baseline_characters": baseline_characters,
        "baseline_score": baseline_score,
        "baseline_lao_ratio": baseline_lao_ratio,
        "probe_below_confidence": DEFAULT_AUTO_ORIENT_PROBE_BELOW_CONFIDENCE,
        "lao_dominant_min_confidence": (
            DEFAULT_AUTO_ORIENT_LAO_DOMINANT_MIN_CONFIDENCE
        ),
        "lao_dominant_min_characters": (
            DEFAULT_AUTO_ORIENT_LAO_DOMINANT_MIN_CHARACTERS
        ),
        "lao_dominant_ratio": DEFAULT_AUTO_ORIENT_LAO_DOMINANT_RATIO,
    }

    if probe_skip_reason is not None:
        return (
            image,
            baseline_lines,
            0,
            {
                **base_diagnostics,
                "selected_confidence": baseline_confidence,
                "selected_characters": baseline_characters,
                "selected_score": baseline_score,
                "selected_lao_ratio": baseline_lao_ratio,
                "probe_skipped": True,
                "probe_skip_reason": probe_skip_reason,
                "probe_strategy": "skipped",
                "probed_degrees": [],
                "engine_orientation_hint": None,
                "best_degrees_clockwise": 0,
                "best_score": baseline_score,
                "runner_up_degrees_clockwise": None,
                "runner_up_score": None,
                "best_score_margin_ratio": None,
            },
        )

    candidates = [
        (
            0,
            image,
            baseline_lines,
            baseline_confidence,
            baseline_characters,
            baseline_score,
            baseline_lao_ratio,
        )
    ]
    probed_degrees: list[int] = []

    def probe(degrees: int):
        _raise_if_cancelled(should_cancel)
        rotated = _rotate_image_clockwise(image, degrees)
        try:
            lines = engine.recognize(rotated)
        except OcrEngineError:
            return None
        confidence, characters, score, lao_ratio = _orientation_line_stats(lines)
        candidate = (
            degrees,
            rotated,
            lines,
            confidence,
            characters,
            score,
            lao_ratio,
        )
        candidates.append(candidate)
        probed_degrees.append(degrees)
        return candidate

    for degrees in (90, 180, 270):
        probe(degrees)

    ranked_candidates = sorted(
        candidates,
        key=lambda item: (item[5], item[3], item[4]),
        reverse=True,
    )
    best = ranked_candidates[0]
    runner_up = ranked_candidates[1] if len(ranked_candidates) > 1 else None
    (
        best_degrees,
        best_image,
        best_lines,
        best_confidence,
        best_characters,
        best_score,
        best_lao_ratio,
    ) = best

    runner_up_degrees = int(runner_up[0]) if runner_up is not None else None
    runner_up_score = float(runner_up[5]) if runner_up is not None else None
    best_score_margin_ratio = (
        (best_score / runner_up_score) - 1.0
        if runner_up_score is not None and runner_up_score > 0
        else None
    )

    use_best = _orientation_candidate_is_acceptable(
        degrees=best_degrees,
        confidence=best_confidence,
        characters=best_characters,
        score=best_score,
        baseline_confidence=baseline_confidence,
        baseline_characters=baseline_characters,
        baseline_score=baseline_score,
    )

    if not use_best:
        best_degrees = 0
        best_image = image
        best_lines = baseline_lines
        best_confidence = baseline_confidence
        best_characters = baseline_characters
        best_score = baseline_score
        best_lao_ratio = baseline_lao_ratio

    return (
        best_image,
        best_lines,
        int(best_degrees),
        {
            **base_diagnostics,
            "selected_confidence": best_confidence,
            "selected_characters": best_characters,
            "selected_score": best_score,
            "selected_lao_ratio": best_lao_ratio,
            "probe_skipped": False,
            "probe_skip_reason": None,
            "probe_strategy": "exhaustive",
            "probed_degrees": list(probed_degrees),
            "engine_orientation_hint": None,
            "best_degrees_clockwise": int(best[0]),
            "best_score": float(best[5]),
            "runner_up_degrees_clockwise": runner_up_degrees,
            "runner_up_score": runner_up_score,
            "best_score_margin_ratio": best_score_margin_ratio,
        },
    )

def process_document(
    path: str | Path,
    *,
    source_name: str | None = None,
    engine: OcrEngine | None = None,
    max_pages: int = 60,
    max_page_pixels: int = DEFAULT_MAX_PAGE_PIXELS,
    should_cancel: Callable[[], bool] | None = None,
    reading_order_resolver: ReadingOrderResolver | None = None,
    auto_orient_right_angles: bool = False,
    include_ocr_line_stats: bool = False,
) -> Document:
    path = Path(path)
    engine = engine or TesseractEngine()
    resolver = reading_order_resolver or DeterministicReadingOrderResolver()
    _raise_if_cancelled(should_cancel)
    if max_page_pixels < 1:
        raise ValueError("max_page_pixels must be at least 1")
    pages = _load_pages(
        path,
        max_pages=max_pages,
        max_page_pixels=max_page_pixels,
        should_cancel=should_cancel,
    )

    output_pages: list[Page] = []
    orientation_pages: list[dict[str, object]] = []
    ocr_line_stats_pages: list[dict[str, object]] = []
    for page_number, loaded_page in enumerate(pages, start=1):
        _raise_if_cancelled(should_cancel)
        cleaned = preprocess_image(loaded_page.image)
        source_image = loaded_page.image
        embedded_images = loaded_page.embedded_images
        orientation_degrees = 0
        orientation_diagnostics: dict[str, object] | None = None
        try:
            if auto_orient_right_angles:
                (
                    cleaned,
                    lines,
                    orientation_degrees,
                    orientation_diagnostics,
                ) = _recognize_with_right_angle_orientation(
                    cleaned,
                    engine=engine,
                    should_cancel=should_cancel,
                )
                if orientation_degrees:
                    original_width = source_image.width
                    original_height = source_image.height
                    source_image = _rotate_image_clockwise(
                        source_image,
                        orientation_degrees,
                    )
                    embedded_images = _rotate_embedded_assets(
                        embedded_images,
                        page_width=original_width,
                        page_height=original_height,
                        degrees=orientation_degrees,
                    )
            else:
                lines = engine.recognize(cleaned)
        except OcrEngineError as exc:
            raise DocumentProcessingError(str(exc)) from exc

        orientation_pages.append(
            {
                "page": page_number,
                "degrees_clockwise": orientation_degrees,
                "diagnostics": orientation_diagnostics,
            }
        )
        if include_ocr_line_stats:
            (
                line_mean_confidence,
                line_recognized_characters,
                line_score,
                line_lao_ratio,
            ) = _orientation_line_stats(lines)
            ocr_line_stats_pages.append(
                {
                    "page": page_number,
                    "mean_confidence": line_mean_confidence,
                    "recognized_characters": line_recognized_characters,
                    "score": line_score,
                    "lao_ratio": line_lao_ratio,
                }
            )

        semantic_blocks = build_page_blocks(lines, cleaned)
        table_boxes = [
            block.bbox
            for block in semantic_blocks
            if block.type == BlockType.TABLE and block.bbox is not None
        ]
        embedded_blocks = [asset.to_block() for asset in embedded_images]
        embedded_boxes = [asset.bbox for asset in embedded_images]
        line_boxes = [line.bbox for line in lines]

        learned_visual_blocks = engine.visual_blocks(
            cleaned,
            source_image=source_image,
            exclude_boxes=[
                *table_boxes,
                *embedded_boxes,
                *line_boxes,
            ],
        )
        learned_visual_boxes = [
            block.bbox
            for block in learned_visual_blocks
            if block.bbox is not None
        ]

        raster_blocks = detect_raster_regions(
            cleaned,
            lines,
            source_image=source_image,
            exclude_boxes=[
                *table_boxes,
                *embedded_boxes,
                *learned_visual_boxes,
            ],
        )
        raster_boxes = [
            block.bbox
            for block in raster_blocks
            if block.bbox is not None
        ]

        diagram_blocks = detect_diagram_regions(
            cleaned,
            lines,
            source_image=source_image,
            exclude_boxes=[
                *table_boxes,
                *embedded_boxes,
                *learned_visual_boxes,
                *raster_boxes,
            ],
        )

        blocks = [
            *semantic_blocks,
            *embedded_blocks,
            *learned_visual_blocks,
            *raster_blocks,
            *diagram_blocks,
        ]
        blocks = resolver.order(
            blocks,
            page_width=cleaned.width,
            page_height=cleaned.height,
        )

        output_pages.append(
            Page(
                number=page_number,
                width=cleaned.width,
                height=cleaned.height,
                blocks=blocks,
            )
        )

    _raise_if_cancelled(should_cancel)
    mark_repeated_headers_footers(output_pages)

    metadata: dict[str, object] = {
        "engine": engine.metadata(),
        "reading_order": resolver.metadata(),
        "page_count": len(output_pages),
        "auto_orientation": {
            "enabled": auto_orient_right_angles,
            "probe_below_confidence": DEFAULT_AUTO_ORIENT_PROBE_BELOW_CONFIDENCE,
            "lao_dominant_min_confidence": (
                DEFAULT_AUTO_ORIENT_LAO_DOMINANT_MIN_CONFIDENCE
            ),
            "lao_dominant_min_characters": (
                DEFAULT_AUTO_ORIENT_LAO_DOMINANT_MIN_CHARACTERS
            ),
            "lao_dominant_ratio": DEFAULT_AUTO_ORIENT_LAO_DOMINANT_RATIO,
            "pages": orientation_pages,
        },
    }
    if include_ocr_line_stats:
        metadata["ocr_line_stats"] = {"pages": ocr_line_stats_pages}

    return Document(
        source_name=source_name or path.name,
        pages=output_pages,
        metadata=metadata,
    )
