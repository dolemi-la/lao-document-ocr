from __future__ import annotations

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
from lao_document_ocr.reading_order import order_blocks
from lao_document_ocr.structure import build_page_blocks

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
SUPPORTED_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | {".pdf"}


class DocumentProcessingError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedPage:
    image: Image.Image
    embedded_images: tuple[EmbeddedImageAsset, ...] = ()


def _render_pdf(path: Path, max_pages: int) -> list[LoadedPage]:
    pages: list[LoadedPage] = []
    try:
        pdf = pymupdf.open(path)
    except Exception as exc:
        raise DocumentProcessingError(f"Could not open PDF: {exc}") from exc

    try:
        if pdf.page_count > max_pages:
            raise DocumentProcessingError(
                f"PDF has {pdf.page_count} pages; maximum is {max_pages}."
            )
        for page in pdf:
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
            mode = "RGB" if pixmap.n < 4 else "RGBA"
            image = Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)
            embedded = extract_pdf_embedded_images(
                pdf,
                page,
                rendered_width=pixmap.width,
                rendered_height=pixmap.height,
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


def _load_pages(path: Path, max_pages: int) -> list[LoadedPage]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _render_pdf(path, max_pages)
    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        try:
            with Image.open(path) as image:
                return [LoadedPage(image=image.convert("RGB"))]
        except Exception as exc:
            raise DocumentProcessingError(f"Could not open image: {exc}") from exc
    raise DocumentProcessingError(f"Unsupported file type: {suffix or 'unknown'}")


def process_document(
    path: str | Path,
    *,
    source_name: str | None = None,
    engine: OcrEngine | None = None,
    max_pages: int = 60,
) -> Document:
    path = Path(path)
    engine = engine or TesseractEngine()
    pages = _load_pages(path, max_pages=max_pages)

    output_pages: list[Page] = []
    for page_number, loaded_page in enumerate(pages, start=1):
        cleaned = preprocess_image(loaded_page.image)
        try:
            lines = engine.recognize(cleaned)
        except OcrEngineError as exc:
            raise DocumentProcessingError(str(exc)) from exc

        blocks = build_page_blocks(lines, cleaned)
        blocks.extend(asset.to_block() for asset in loaded_page.embedded_images)
        blocks = order_blocks(blocks, page_width=cleaned.width)

        output_pages.append(
            Page(
                number=page_number,
                width=cleaned.width,
                height=cleaned.height,
                blocks=blocks,
            )
        )

    mark_repeated_headers_footers(output_pages)

    return Document(
        source_name=source_name or path.name,
        pages=output_pages,
        metadata={
            "engine": engine.metadata(),
            "page_count": len(output_pages),
        },
    )
