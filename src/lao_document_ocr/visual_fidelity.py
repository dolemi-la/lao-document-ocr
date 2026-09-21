from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageOps


@dataclass(frozen=True)
class PageFidelityMetrics:
    page: int
    pixel_similarity: float
    foreground_iou: float
    edge_f1: float
    composite_score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DocumentFidelityMetrics:
    reference_pages: int
    predicted_pages: int
    compared_pages: int
    page_count_score: float
    pixel_similarity: float
    foreground_iou: float
    edge_f1: float
    composite_score: float
    pages: tuple[PageFidelityMetrics, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["pages"] = [page.to_dict() for page in self.pages]
        return payload


def find_office_binary(explicit: str | Path | None = None) -> Path | None:
    if explicit is not None:
        candidate = Path(explicit)
        return candidate if candidate.is_file() else None

    for name in ("libreoffice", "soffice", "loffice"):
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)
    return None


def office_version(binary: str | Path) -> str | None:
    try:
        completed = subprocess.run(
            [str(binary), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    value = (completed.stdout or completed.stderr).strip()
    return value or None


def render_pdf_pages(path: str | Path, *, dpi: int = 144) -> list[Image.Image]:
    if dpi < 72:
        raise ValueError("dpi must be at least 72")

    source = Path(path)
    try:
        pdf = pymupdf.open(source)
    except Exception as exc:
        raise ValueError(f"Could not open PDF for rendering: {exc}") from exc

    scale = dpi / 72.0
    pages: list[Image.Image] = []
    try:
        for page in pdf:
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(scale, scale),
                alpha=False,
            )
            image = Image.frombytes(
                "RGB",
                (pixmap.width, pixmap.height),
                pixmap.samples,
            )
            pages.append(image)
    finally:
        pdf.close()
    return pages


def load_reference_pages(path: str | Path, *, dpi: int = 144) -> list[Image.Image]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Reference document not found: {source}")

    if source.suffix.lower() == ".pdf":
        return render_pdf_pages(source, dpi=dpi)

    try:
        with Image.open(source) as image:
            return [ImageOps.exif_transpose(image).convert("RGB")]
    except Exception as exc:
        raise ValueError(f"Could not open reference image: {exc}") from exc


def render_docx_pages(
    path: str | Path,
    *,
    dpi: int = 144,
    office_binary: str | Path | None = None,
    timeout_seconds: int = 120,
) -> tuple[list[Image.Image], Path]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"DOCX not found: {source}")

    binary = find_office_binary(office_binary)
    if binary is None:
        raise RuntimeError(
            "LibreOffice/soffice was not found. Install LibreOffice or pass "
            "--office-binary to benchmark rendered DOCX output."
        )

    with tempfile.TemporaryDirectory(prefix="lao-ocr-docx-render-") as temp_dir:
        output_dir = Path(temp_dir)
        try:
            completed = subprocess.run(
                [
                    str(binary),
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(output_dir),
                    str(source),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("DOCX rendering timed out") from exc

        pdf_path = output_dir / f"{source.stem}.pdf"
        if completed.returncode != 0 or not pdf_path.is_file():
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            detail = stderr or stdout or f"exit code {completed.returncode}"
            raise RuntimeError(f"LibreOffice DOCX rendering failed: {detail}")

        pages = render_pdf_pages(pdf_path, dpi=dpi)
        # The temporary PDF disappears after this function, so return the
        # renderer binary for report metadata rather than the PDF path.
        return pages, binary


def _fit_to_reference(
    prediction: Image.Image,
    reference_size: tuple[int, int],
) -> Image.Image:
    target_width, target_height = reference_size
    fitted = ImageOps.contain(
        prediction.convert("RGB"),
        reference_size,
        method=Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", reference_size, "white")
    left = (target_width - fitted.width) // 2
    top = (target_height - fitted.height) // 2
    canvas.paste(fitted, (left, top))
    return canvas


def _foreground_iou(reference_gray: np.ndarray, prediction_gray: np.ndarray) -> float:
    reference = reference_gray < 245
    prediction = prediction_gray < 245
    union = np.logical_or(reference, prediction)
    if not np.any(union):
        return 1.0
    intersection = np.logical_and(reference, prediction)
    return float(np.count_nonzero(intersection) / np.count_nonzero(union))


def _edge_f1(reference_gray: np.ndarray, prediction_gray: np.ndarray) -> float:
    reference_edges = cv2.Canny(reference_gray, 70, 160) > 0
    prediction_edges = cv2.Canny(prediction_gray, 70, 160) > 0
    if not np.any(reference_edges) and not np.any(prediction_edges):
        return 1.0
    if not np.any(reference_edges) or not np.any(prediction_edges):
        return 0.0

    kernel = np.ones((3, 3), dtype=np.uint8)
    reference_dilated = cv2.dilate(reference_edges.astype(np.uint8), kernel) > 0
    prediction_dilated = cv2.dilate(prediction_edges.astype(np.uint8), kernel) > 0

    precision = float(
        np.count_nonzero(prediction_edges & reference_dilated)
        / np.count_nonzero(prediction_edges)
    )
    recall = float(
        np.count_nonzero(reference_edges & prediction_dilated)
        / np.count_nonzero(reference_edges)
    )
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compare_page_images(
    reference: Image.Image,
    prediction: Image.Image,
    *,
    page_number: int = 1,
) -> PageFidelityMetrics:
    reference_rgb = reference.convert("RGB")
    prediction_rgb = _fit_to_reference(prediction, reference_rgb.size)

    reference_gray = np.asarray(
        reference_rgb.convert("L"),
        dtype=np.uint8,
    )
    prediction_gray = np.asarray(
        prediction_rgb.convert("L"),
        dtype=np.uint8,
    )

    mean_absolute_error = float(
        np.mean(
            np.abs(
                reference_gray.astype(np.int16)
                - prediction_gray.astype(np.int16)
            )
        )
    )
    pixel_similarity = max(0.0, 1.0 - mean_absolute_error / 255.0)
    foreground_iou = _foreground_iou(reference_gray, prediction_gray)
    edge_f1 = _edge_f1(reference_gray, prediction_gray)
    composite = (
        0.4 * pixel_similarity
        + 0.3 * foreground_iou
        + 0.3 * edge_f1
    )

    return PageFidelityMetrics(
        page=page_number,
        pixel_similarity=pixel_similarity,
        foreground_iou=foreground_iou,
        edge_f1=edge_f1,
        composite_score=composite,
    )


def compare_document_images(
    reference_pages: list[Image.Image],
    prediction_pages: list[Image.Image],
) -> DocumentFidelityMetrics:
    if not reference_pages:
        raise ValueError("Reference document has no pages")
    if not prediction_pages:
        raise ValueError("Predicted document has no pages")

    compared = min(len(reference_pages), len(prediction_pages))
    page_metrics = tuple(
        compare_page_images(
            reference_pages[index],
            prediction_pages[index],
            page_number=index + 1,
        )
        for index in range(compared)
    )

    page_count_score = min(len(reference_pages), len(prediction_pages)) / max(
        len(reference_pages),
        len(prediction_pages),
    )
    pixel_similarity = sum(page.pixel_similarity for page in page_metrics) / compared
    foreground_iou = sum(page.foreground_iou for page in page_metrics) / compared
    edge_f1 = sum(page.edge_f1 for page in page_metrics) / compared
    base_composite = sum(page.composite_score for page in page_metrics) / compared

    return DocumentFidelityMetrics(
        reference_pages=len(reference_pages),
        predicted_pages=len(prediction_pages),
        compared_pages=compared,
        page_count_score=page_count_score,
        pixel_similarity=pixel_similarity,
        foreground_iou=foreground_iou,
        edge_f1=edge_f1,
        composite_score=base_composite * page_count_score,
        pages=page_metrics,
    )


def benchmark_docx_fidelity(
    reference_path: str | Path,
    docx_path: str | Path,
    *,
    dpi: int = 144,
    office_binary: str | Path | None = None,
) -> dict:
    reference_pages = load_reference_pages(reference_path, dpi=dpi)
    prediction_pages, renderer = render_docx_pages(
        docx_path,
        dpi=dpi,
        office_binary=office_binary,
    )
    metrics = compare_document_images(reference_pages, prediction_pages)
    return {
        "schema_version": "1",
        "generated_at": datetime.now(UTC).isoformat(),
        "dpi": dpi,
        "renderer": {
            "binary": str(renderer),
            "version": office_version(renderer),
            "pymupdf_version": getattr(pymupdf, "VersionBind", None),
        },
        "metrics": metrics.to_dict(),
    }


def write_fidelity_report(report: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
