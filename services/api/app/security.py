from __future__ import annotations

import math
import unicodedata
from pathlib import Path

import pymupdf
from PIL import Image


class UploadValidationError(ValueError):
    pass


_IMAGE_FORMATS = {
    ".png": {"PNG"},
    ".jpg": {"JPEG"},
    ".jpeg": {"JPEG"},
    ".tif": {"TIFF"},
    ".tiff": {"TIFF"},
    ".webp": {"WEBP"},
}


def sanitize_filename(filename: str | None, *, max_stem_length: int = 120) -> str:
    raw = unicodedata.normalize("NFC", filename or "document")
    raw = raw.replace("\\", "/")
    name = raw.rsplit("/", 1)[-1].strip()
    if not name or name in {".", ".."}:
        return "document"

    suffix = Path(name).suffix.lower()
    stem = name[: -len(suffix)] if suffix else name

    safe_stem = "".join(
        char
        if (
            char.isalnum()
            or unicodedata.category(char).startswith("M")
            or char in {" ", ".", "_", "-"}
        )
        else "_"
        for char in stem
        if not unicodedata.category(char).startswith("C")
    )
    safe_stem = " ".join(safe_stem.split()).strip(" .")
    safe_stem = safe_stem[:max_stem_length].rstrip(" .")
    if not safe_stem:
        safe_stem = "document"

    safe_suffix = "".join(
        char if char.isalnum() or char in {".", "_", "-"} else "_"
        for char in suffix
        if not unicodedata.category(char).startswith("C")
    )
    return f"{safe_stem}{safe_suffix}"


def _validate_pdf(
    path: Path,
    *,
    max_pages: int,
    max_page_pixels: int,
) -> None:
    with path.open("rb") as source:
        if source.read(5) != b"%PDF-":
            raise UploadValidationError("File content does not match the .pdf extension.")

    try:
        pdf = pymupdf.open(path)
    except Exception as exc:
        raise UploadValidationError("Invalid PDF file.") from exc

    try:
        if pdf.needs_pass:
            raise UploadValidationError("Encrypted PDFs are not supported.")
        if pdf.page_count < 1:
            raise UploadValidationError("PDF contains no pages.")
        if pdf.page_count > max_pages:
            raise UploadValidationError(
                f"PDF has {pdf.page_count} pages; maximum is {max_pages}."
            )
        for page in pdf:
            width = math.ceil(float(page.rect.width) * 2)
            height = math.ceil(float(page.rect.height) * 2)
            if width * height > max_page_pixels:
                raise UploadValidationError(
                    f"PDF page {page.number + 1} exceeds the rendered pixel limit."
                )
    finally:
        pdf.close()


def _validate_image(
    path: Path,
    suffix: str,
    *,
    max_page_pixels: int,
) -> None:
    expected = _IMAGE_FORMATS[suffix]
    try:
        with Image.open(path) as image:
            actual = (image.format or "").upper()
            if actual not in expected:
                raise UploadValidationError(
                    f"File content does not match the {suffix} extension."
                )
            if image.width * image.height > max_page_pixels:
                raise UploadValidationError(
                    "Image dimensions exceed the pixel limit."
                )
            image.verify()
    except UploadValidationError:
        raise
    except Exception as exc:
        raise UploadValidationError("Invalid image file.") from exc


def validate_uploaded_content(
    path: str | Path,
    suffix: str,
    *,
    max_pages: int,
    max_page_pixels: int,
) -> None:
    source = Path(path)
    if not source.is_file() or source.stat().st_size == 0:
        raise UploadValidationError("Uploaded file is empty.")
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")
    if max_page_pixels < 1:
        raise ValueError("max_page_pixels must be at least 1")

    normalized_suffix = suffix.lower()
    if normalized_suffix == ".pdf":
        _validate_pdf(
            source,
            max_pages=max_pages,
            max_page_pixels=max_page_pixels,
        )
        return
    if normalized_suffix in _IMAGE_FORMATS:
        _validate_image(
            source,
            normalized_suffix,
            max_page_pixels=max_page_pixels,
        )
        return
    raise UploadValidationError("Unsupported uploaded file type.")
