from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image

from lao_document_ocr.embedded_images import EmbeddedImageAsset, extract_pdf_embedded_images
from lao_document_ocr.header_footer import mark_repeated_headers_footers
from lao_document_ocr.models import Document, Page
from lao_document_ocr.ocr.base import OcrEngine, OcrEngineError
from lao_document_ocr.ocr.tesseract import TesseractEngine
from lao_document_ocr.preprocessing import preprocess_image
from lao_document_ocr.raster_regions import detect_raster_regions
from lao_document_ocr.reading_order import order_blocks
from lao_document_ocr.structure import build_page_blocks

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
SUPPORTED_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | {".pdf"}
DEFAULT_MAX_PAGE_PIXELS = 40_000_000


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


def process_document(
    path: str | Path,
    *,
    source_name: str | None = None,
    engine: OcrEngine | None = None,
    max_pages: int = 60,
    max_page_pixels: int = DEFAULT_MAX_PAGE_PIXELS,
    should_cancel: Callable[[], bool] | None = None,
) -> Document:
    path = Path(path)
    engine = engine or TesseractEngine()
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
    for page_number, loaded_page in enumerate(pages, start=1):
        _raise_if_cancelled(should_cancel)
        cleaned = preprocess_image(loaded_page.image)
        try:
            lines = engine.recognize(cleaned)
        except OcrEngineError as exc:
            raise DocumentProcessingError(str(exc)) from exc

        blocks = build_page_blocks(lines, cleaned)
        blocks.extend(asset.to_block() for asset in loaded_page.embedded_images)
        blocks.extend(
            detect_raster_regions(
                cleaned,
                lines,
                source_image=loaded_page.image,
                exclude_boxes=[asset.bbox for asset in loaded_page.embedded_images],
            )
        )
        blocks = order_blocks(blocks, page_width=cleaned.width)

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

    return Document(
        source_name=source_name or path.name,
        pages=output_pages,
        metadata={
            "engine": engine.metadata(),
            "page_count": len(output_pages),
        },
    )
