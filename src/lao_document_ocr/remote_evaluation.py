from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import platform
import socket
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pymupdf
from PIL import Image

from lao_document_ocr.models import BlockType
from lao_document_ocr.ocr.base import OcrEngine
from lao_document_ocr.pipeline import DEFAULT_MAX_PAGE_PIXELS, process_document
from lao_document_ocr.reading_order import ReadingOrderResolver

_REMOTE_STATUS_PREFIX = "remote-evaluation-"
_USER_AGENT = (
    "lao-document-ocr-remote-eval/0.1 "
    "(https://github.com/dolemi-la/lao-document-ocr)"
)
_CHUNK_SIZE = 1024 * 1024


class RemoteEvaluationError(ValueError):
    pass


@dataclass(frozen=True)
class DownloadedRemote:
    path: Path
    final_url: str
    sha256: str
    size_bytes: int
    content_type: str | None
    etag: str | None
    last_modified: str | None
    format: str


RemoteFetcher = Callable[..., DownloadedRemote]


def _is_blocked_ip(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_remote_url(url: str) -> urllib.parse.SplitResult:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() != "https":
        raise RemoteEvaluationError("Remote evaluation URLs must use HTTPS.")
    if not parsed.hostname:
        raise RemoteEvaluationError("Remote evaluation URL is missing a hostname.")
    if parsed.username is not None or parsed.password is not None:
        raise RemoteEvaluationError("Remote evaluation URLs must not contain user info.")
    if parsed.port not in (None, 443):
        raise RemoteEvaluationError("Remote evaluation URLs may only use HTTPS port 443.")

    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(
                parsed.hostname,
                parsed.port or 443,
                type=socket.SOCK_STREAM,
            )
        }
    except OSError as exc:
        raise RemoteEvaluationError(
            f"Could not resolve remote host: {parsed.hostname}"
        ) from exc

    if not addresses:
        raise RemoteEvaluationError(
            f"Remote host resolved to no addresses: {parsed.hostname}"
        )
    blocked = sorted(address for address in addresses if _is_blocked_ip(address))
    if blocked:
        raise RemoteEvaluationError(
            f"Remote host resolves to a non-public address: {blocked[0]}"
        )
    return parsed


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req,
        fp,
        code,
        msg,
        headers,
        newurl,
    ):
        validate_remote_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _detect_format(prefix: bytes) -> tuple[str, str]:
    if prefix.startswith(b"%PDF-"):
        return "pdf", ".pdf"
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png", ".png"
    if prefix.startswith(b"\xff\xd8\xff"):
        return "jpeg", ".jpg"
    if prefix.startswith((b"II*\x00", b"MM\x00*")):
        return "tiff", ".tiff"
    if len(prefix) >= 12 and prefix[:4] == b"RIFF" and prefix[8:12] == b"WEBP":
        return "webp", ".webp"
    raise RemoteEvaluationError(
        "Remote source is not a supported PDF/PNG/JPEG/TIFF/WebP document."
    )


def download_remote_source(
    url: str,
    destination_base: str | Path,
    *,
    max_bytes: int = 25 * 1024 * 1024,
    timeout_seconds: float = 20.0,
) -> DownloadedRemote:
    if max_bytes < 1:
        raise ValueError("max_bytes must be at least 1")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    validate_remote_url(url)
    opener = urllib.request.build_opener(_SafeRedirectHandler())
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/pdf,image/*;q=0.9,*/*;q=0.1",
        },
    )

    base = Path(destination_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    temp_path = base.with_suffix(".download")
    digest = hashlib.sha256()
    size = 0
    prefix = b""

    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            final_url = response.geturl()
            validate_remote_url(final_url)

            length_header = response.headers.get("Content-Length")
            if length_header:
                try:
                    content_length = int(length_header)
                except ValueError:
                    content_length = None
                if content_length is not None and content_length > max_bytes:
                    raise RemoteEvaluationError(
                        f"Remote source exceeds download limit ({content_length} > {max_bytes})."
                    )

            with temp_path.open("wb") as handle:
                while True:
                    chunk = response.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        raise RemoteEvaluationError(
                            f"Remote source exceeds download limit ({max_bytes} bytes)."
                        )
                    if len(prefix) < 16:
                        prefix = (prefix + chunk)[:16]
                    digest.update(chunk)
                    handle.write(chunk)

            doc_format, suffix = _detect_format(prefix)
            final_path = base.with_suffix(suffix)
            os.replace(temp_path, final_path)
            return DownloadedRemote(
                path=final_path,
                final_url=final_url,
                sha256=digest.hexdigest(),
                size_bytes=size,
                content_type=response.headers.get_content_type(),
                etag=response.headers.get("ETag"),
                last_modified=response.headers.get("Last-Modified"),
                format=doc_format,
            )
    except RemoteEvaluationError:
        temp_path.unlink(missing_ok=True)
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        temp_path.unlink(missing_ok=True)
        raise RemoteEvaluationError(f"Remote download failed: {exc}") from exc


def load_remote_registry_sources(
    registry_path: str | Path,
    *,
    source_ids: Iterable[str] | None = None,
    all_remote: bool = False,
) -> list[dict[str, Any]]:
    path = Path(registry_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RemoteEvaluationError(f"Could not read source registry: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RemoteEvaluationError(f"Invalid source registry JSON: {exc}") from exc

    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list):
        raise RemoteEvaluationError("Source registry contains no sources list.")

    by_id = {
        str(source.get("id")): source
        for source in raw_sources
        if isinstance(source, dict) and source.get("id")
    }

    requested = list(dict.fromkeys(source_ids or []))
    if all_remote and requested:
        raise RemoteEvaluationError(
            "Choose either explicit source_ids or all_remote, not both."
        )
    if not all_remote and not requested:
        raise RemoteEvaluationError(
            "Remote evaluation requires explicit source IDs or all_remote=True."
        )

    if all_remote:
        selected = [
            source
            for source in raw_sources
            if isinstance(source, dict)
            and str(source.get("status", "")).startswith(_REMOTE_STATUS_PREFIX)
        ]
    else:
        selected = []
        for source_id in requested:
            source = by_id.get(source_id)
            if source is None:
                raise RemoteEvaluationError(
                    f"Source registry has no source id: {source_id}"
                )
            status = str(source.get("status", ""))
            if not status.startswith(_REMOTE_STATUS_PREFIX):
                raise RemoteEvaluationError(
                    f"Source is not a remote-evaluation entry: {source_id}"
                )
            selected.append(source)

    if not selected:
        raise RemoteEvaluationError("No remote evaluation sources selected.")
    return selected


def _text_stats(text: str) -> dict[str, int | float]:
    nonspace = [char for char in text if not char.isspace()]
    lao = sum(1 for char in nonspace if "\u0e80" <= char <= "\u0eff")
    latin = 0
    other_letters = 0
    marks = 0
    punctuation_symbols = 0
    for char in nonspace:
        category = unicodedata.category(char)
        name = unicodedata.name(char, "")
        if category.startswith("L"):
            if "\u0e80" <= char <= "\u0eff":
                continue
            if "LATIN" in name:
                latin += 1
            else:
                other_letters += 1
        elif category.startswith("M"):
            marks += 1
        elif category.startswith(("P", "S")):
            punctuation_symbols += 1

    digits = sum(1 for char in nonspace if char.isdigit())
    replacement = text.count("\ufffd")
    total = len(nonspace)
    letter_total = lao + latin + other_letters
    return {
        "characters": len(text),
        "nonspace_characters": total,
        "lines": len([line for line in text.splitlines() if line.strip()]),
        "words": len(text.split()),
        "lao_characters": lao,
        "latin_characters": latin,
        "other_letter_characters": other_letters,
        "letter_characters": letter_total,
        "mark_characters": marks,
        "punctuation_symbol_characters": punctuation_symbols,
        "digit_characters": digits,
        "replacement_characters": replacement,
        "lao_ratio": (lao / total) if total else 0.0,
        "other_letter_ratio": (
            (other_letters / letter_total) if letter_total else 0.0
        ),
    }


def _layer_gap_diagnostics(
    native_text: dict[str, int | float],
    ocr_text: dict[str, int | float],
) -> dict[str, Any]:
    native_nonspace = int(native_text["nonspace_characters"])
    ocr_nonspace = int(ocr_text["nonspace_characters"])
    native_lao = int(native_text["lao_characters"])
    ocr_lao = int(ocr_text["lao_characters"])
    native_latin = int(native_text.get("latin_characters", 0))
    ocr_latin = int(ocr_text.get("latin_characters", 0))
    native_other = int(native_text.get("other_letter_characters", 0))
    ocr_other = int(ocr_text.get("other_letter_characters", 0))
    native_letters = int(
        native_text.get(
            "letter_characters",
            native_lao + native_latin + native_other,
        )
    )

    if native_nonspace == 0 and ocr_nonspace > 0:
        classification = "native-layer-empty"
    elif native_nonspace <= 32 and ocr_nonspace >= max(100, native_nonspace * 5):
        classification = "native-layer-sparse"
    elif (
        native_nonspace >= 100
        and native_other >= max(20, int(native_letters * 0.15))
        and ocr_other <= max(5, int(native_other * 0.1))
        and (ocr_lao + ocr_latin) >= 50
    ):
        classification = "native-layer-script-anomaly"
    elif ocr_lao >= 50 and native_lao < max(5, int(ocr_lao * 0.1)):
        classification = "lao-missing-from-native-layer"
    elif native_nonspace > 0 and ocr_nonspace >= native_nonspace * 2:
        classification = "ocr-much-richer-than-native"
    else:
        classification = "no-large-gap-detected"

    return {
        "classification": classification,
        "native_layer_empty": native_nonspace == 0,
        "native_layer_sparse": native_nonspace <= 32,
        "ocr_minus_native_nonspace_characters": ocr_nonspace - native_nonspace,
        "ocr_minus_native_lao_characters": ocr_lao - native_lao,
        "ocr_to_native_nonspace_ratio": (
            (ocr_nonspace / native_nonspace) if native_nonspace else None
        ),
        "native_other_letter_characters": native_other,
        "ocr_other_letter_characters": ocr_other,
    }


def _confidence_diagnostics(mean_confidence: float | None) -> dict[str, Any]:
    if mean_confidence is None:
        band = "no-confidence"
    elif mean_confidence < 0.60:
        band = "low"
    elif mean_confidence < 0.80:
        band = "medium"
    else:
        band = "high"
    return {
        "band": band,
        "mean_block_confidence": mean_confidence,
    }


def _ocr_document_stats(document) -> dict[str, Any]:
    block_counts = Counter()
    confidences: list[float] = []
    for page in document.pages:
        for block in page.blocks:
            block_counts[block.type.value] += 1
            if block.confidence is not None:
                confidences.append(float(block.confidence))

    line_stats = None
    line_stats_payload = document.metadata.get("ocr_line_stats")
    if isinstance(line_stats_payload, dict):
        pages = line_stats_payload.get("pages")
        if isinstance(pages, list) and len(pages) == 1 and isinstance(pages[0], dict):
            line_stats = dict(pages[0])

    stats = {
        "text": _text_stats(document.plain_text),
        "blocks": {
            block_type.value: block_counts.get(block_type.value, 0)
            for block_type in BlockType
        },
        "mean_block_confidence": (
            sum(confidences) / len(confidences)
            if confidences
            else None
        ),
        "confidence_samples": len(confidences),
    }
    if line_stats is not None:
        stats["line_stats"] = line_stats
    return stats


def _rotation_probe_metrics(
    ocr_stats: dict[str, Any],
) -> tuple[float, int, float]:
    line_stats = ocr_stats.get("line_stats")
    if isinstance(line_stats, dict):
        confidence = float(line_stats.get("mean_confidence", 0.0))
        characters = max(0, int(line_stats.get("recognized_characters", 0)))
        score = float(line_stats.get("score", 0.0))
        return confidence, characters, score

    confidence_raw = ocr_stats.get("mean_block_confidence")
    confidence = float(confidence_raw) if confidence_raw is not None else 0.0
    text_stats = ocr_stats.get("text") or {}
    characters = max(0, int(text_stats.get("letter_characters", 0)))
    score = confidence * math.log1p(characters)
    return confidence, characters, score


def _rotation_probe(
    page_path: Path,
    *,
    baseline_ocr: dict[str, Any],
    engine: OcrEngine,
    max_page_pixels: int,
    reading_order_resolver: ReadingOrderResolver | None,
) -> dict[str, Any]:
    variants: list[dict[str, Any]] = []
    use_line_stats = isinstance(baseline_ocr.get("line_stats"), dict)

    def append_variant(degrees_clockwise: int, ocr_stats: dict[str, Any]) -> None:
        orientation_confidence, recognized_characters, score = (
            _rotation_probe_metrics(ocr_stats)
        )
        variants.append(
            {
                "degrees_clockwise": degrees_clockwise,
                "score": score,
                "orientation_confidence": orientation_confidence,
                "recognized_characters": recognized_characters,
                "mean_block_confidence": ocr_stats.get("mean_block_confidence"),
                "confidence_band": _confidence_diagnostics(
                    ocr_stats.get("mean_block_confidence")
                )["band"],
                "text": dict(ocr_stats.get("text") or {}),
            }
        )

    append_variant(0, baseline_ocr)
    with Image.open(page_path) as hint_source:
        hint_image = hint_source.convert("RGB")
    orientation_hint = engine.orientation_hint(hint_image)

    transpose = {
        90: Image.Transpose.ROTATE_270,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_90,
    }
    with Image.open(page_path) as source:
        source_rgb = source.convert("RGB")
        for degrees_clockwise, operation in transpose.items():
            rotated = source_rgb.transpose(operation)
            if rotated.width * rotated.height > max_page_pixels:
                continue
            rotated_path = page_path.with_name(
                f"{page_path.stem}-rot{degrees_clockwise}.png"
            )
            rotated.save(rotated_path, format="PNG")
            try:
                document = process_document(
                    rotated_path,
                    source_name=rotated_path.name,
                    engine=engine,
                    max_pages=1,
                    max_page_pixels=max_page_pixels,
                    reading_order_resolver=reading_order_resolver,
                    include_ocr_line_stats=use_line_stats,
                )
                append_variant(
                    degrees_clockwise,
                    _ocr_document_stats(document),
                )
            finally:
                rotated_path.unlink(missing_ok=True)

    ranked = sorted(
        variants,
        key=lambda item: (
            float(item["score"]),
            float(item["orientation_confidence"]),
            int(item["recognized_characters"]),
        ),
        reverse=True,
    )
    best = ranked[0]
    baseline = next(item for item in variants if item["degrees_clockwise"] == 0)
    baseline_score = float(baseline["score"])
    best_score = float(best["score"])
    baseline_confidence = float(baseline["orientation_confidence"])
    best_confidence = float(best["orientation_confidence"])
    baseline_characters = int(baseline["recognized_characters"])
    best_characters = int(best["recognized_characters"])

    recommended = None
    if (
        best["degrees_clockwise"] != 0
        and best_score >= max(0.01, baseline_score * 1.15)
        and best_confidence >= baseline_confidence + 0.08
        and best_characters >= max(20, int(baseline_characters * 0.5))
    ):
        recommended = int(best["degrees_clockwise"])

    return {
        "scoring_basis": (
            "production-line-stats" if use_line_stats else "block-letter-fallback"
        ),
        "variants": variants,
        "best_degrees_clockwise": int(best["degrees_clockwise"]),
        "recommended_degrees_clockwise": recommended,
        "baseline_score": baseline_score,
        "best_score": best_score,
        "score_improvement_ratio": (
            (best_score / baseline_score) if baseline_score > 0 else None
        ),
        "confidence_improvement": best_confidence - baseline_confidence,
        "engine_orientation_hint": orientation_hint,
        "hint_matches_best": (
            (
                int(orientation_hint.get("degrees_clockwise", 0))
                == int(best["degrees_clockwise"])
            )
            if isinstance(orientation_hint, dict)
            else None
        ),
        "hint_matches_recommendation": (
            (
                recommended is not None
                and int(orientation_hint.get("degrees_clockwise", 0))
                == int(recommended)
            )
            if isinstance(orientation_hint, dict)
            else None
        ),
    }


def _sample_page_numbers(
    page_count: int,
    requested_pages: Iterable[int] | None,
    max_pages_per_source: int,
) -> list[int]:
    if page_count < 1:
        raise RemoteEvaluationError("Document contains no pages.")
    if max_pages_per_source < 1:
        raise ValueError("max_pages_per_source must be at least 1")

    requested = list(dict.fromkeys(requested_pages or []))
    if requested:
        invalid = [page for page in requested if page < 1 or page > page_count]
        if invalid:
            raise RemoteEvaluationError(
                f"Requested page is outside document range 1..{page_count}: {invalid[0]}"
            )
        return requested[:max_pages_per_source]

    if max_pages_per_source == 1 or page_count == 1:
        return [1]
    if max_pages_per_source == 2:
        return [1, page_count] if page_count > 1 else [1]

    candidates = [1, math.ceil(page_count / 2), page_count]
    output = list(dict.fromkeys(candidates))
    if len(output) < max_pages_per_source:
        for page in range(1, page_count + 1):
            if page not in output:
                output.append(page)
            if len(output) >= max_pages_per_source:
                break
    return sorted(output[:max_pages_per_source])


def _pdf_page_media_diagnostics(
    page,
    native_text: dict[str, int | float],
) -> dict[str, Any]:
    page_area = float(page.rect.width * page.rect.height) or 1.0
    coverages: list[float] = []
    images = page.get_images(full=True)
    for image in images:
        xref = image[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        for rect in rects:
            coverage = float(rect.width * rect.height) / page_area
            coverages.append(max(0.0, min(1.0, coverage)))

    max_coverage = max(coverages) if coverages else 0.0
    native_nonspace = int(native_text["nonspace_characters"])
    if max_coverage >= 0.90:
        if native_nonspace == 0:
            classification = "full-page-raster-no-text-layer"
        elif native_nonspace <= 32:
            classification = "full-page-raster-sparse-text-layer"
        else:
            classification = "full-page-raster-with-text-overlay"
    elif max_coverage >= 0.20:
        classification = "partial-raster-content"
    else:
        classification = "no-large-raster-layer"

    return {
        "classification": classification,
        "image_count": len(images),
        "max_image_coverage_ratio": max_coverage,
        "full_page_raster": max_coverage >= 0.90,
    }


def _render_pdf_page(
    pdf,
    page_number: int,
    destination: Path,
    *,
    max_page_pixels: int,
) -> tuple[Path, dict[str, Any]]:
    page = pdf.load_page(page_number - 1)
    render_width = math.ceil(float(page.rect.width) * 2)
    render_height = math.ceil(float(page.rect.height) * 2)
    if render_width * render_height > max_page_pixels:
        raise RemoteEvaluationError(
            f"PDF page {page_number} exceeds rendered pixel limit."
        )

    native_text = _text_stats(page.get_text("text"))
    page_media = _pdf_page_media_diagnostics(page, native_text)
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
    mode = "RGB" if pixmap.n < 4 else "RGBA"
    image = Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)
    image.convert("RGB").save(destination, format="PNG")
    return destination, {
        "number": page_number,
        "rotation_degrees": int(page.rotation),
        "native_text": native_text,
        "page_media": page_media,
        "render_width": pixmap.width,
        "render_height": pixmap.height,
    }


def _evaluate_pdf(
    path: Path,
    *,
    engine: OcrEngine,
    requested_pages: Iterable[int] | None,
    max_pages_per_source: int,
    max_document_pages: int,
    max_page_pixels: int,
    work_dir: Path,
    reading_order_resolver: ReadingOrderResolver | None,
    probe_right_angle_rotations: bool,
    auto_orient_right_angles: bool,
) -> dict[str, Any]:
    try:
        pdf = pymupdf.open(path)
    except Exception as exc:
        raise RemoteEvaluationError("Could not open downloaded PDF.") from exc

    try:
        if pdf.needs_pass:
            raise RemoteEvaluationError("Encrypted remote PDFs are not supported.")
        if pdf.page_count > max_document_pages:
            raise RemoteEvaluationError(
                f"Remote PDF has {pdf.page_count} pages; maximum is {max_document_pages}."
            )
        selected = _sample_page_numbers(
            pdf.page_count,
            requested_pages,
            max_pages_per_source,
        )
        page_results = []
        for page_number in selected:
            page_path = work_dir / f"page-{page_number:04d}.png"
            page_path, diagnostics = _render_pdf_page(
                pdf,
                page_number,
                page_path,
                max_page_pixels=max_page_pixels,
            )
            started = time.perf_counter()
            document = process_document(
                page_path,
                source_name=f"{path.name}#page={page_number}",
                engine=engine,
                max_pages=1,
                max_page_pixels=max_page_pixels,
                reading_order_resolver=reading_order_resolver,
                auto_orient_right_angles=auto_orient_right_angles,
                include_ocr_line_stats=(
                    probe_right_angle_rotations and not auto_orient_right_angles
                ),
            )
            diagnostics["ocr"] = _ocr_document_stats(document)
            if auto_orient_right_angles:
                pages_meta = document.metadata.get("auto_orientation", {}).get(
                    "pages", []
                )
                if pages_meta:
                    diagnostics["auto_orientation"] = pages_meta[0]
            diagnostics["layer_gap"] = _layer_gap_diagnostics(
                diagnostics["native_text"],
                diagnostics["ocr"]["text"],
            )
            diagnostics["ocr_quality"] = _confidence_diagnostics(
                diagnostics["ocr"]["mean_block_confidence"]
            )
            if probe_right_angle_rotations and not auto_orient_right_angles:
                diagnostics["rotation_probe"] = _rotation_probe(
                    page_path,
                    baseline_ocr=diagnostics["ocr"],
                    engine=engine,
                    max_page_pixels=max_page_pixels,
                    reading_order_resolver=reading_order_resolver,
                )
            diagnostics["elapsed_seconds"] = time.perf_counter() - started
            page_results.append(diagnostics)

        layer_gap_counts = Counter(
            page["layer_gap"]["classification"] for page in page_results
        )
        return {
            "page_count": pdf.page_count,
            "selected_pages": selected,
            "layer_gap_summary": dict(sorted(layer_gap_counts.items())),
            "pages": page_results,
        }
    finally:
        pdf.close()


def _evaluate_image(
    path: Path,
    *,
    engine: OcrEngine,
    max_page_pixels: int,
    reading_order_resolver: ReadingOrderResolver | None,
    probe_right_angle_rotations: bool,
    auto_orient_right_angles: bool,
) -> dict[str, Any]:
    with Image.open(path) as image:
        if image.width * image.height > max_page_pixels:
            raise RemoteEvaluationError(
                "Remote image dimensions exceed rendered pixel limit."
            )
        dimensions = {
            "number": 1,
            "rotation_degrees": 0,
            "native_text": _text_stats(""),
            "page_media": {
                "classification": "direct-raster-image",
                "image_count": 1,
                "max_image_coverage_ratio": 1.0,
                "full_page_raster": True,
            },
            "render_width": image.width,
            "render_height": image.height,
        }

    started = time.perf_counter()
    document = process_document(
        path,
        source_name=path.name,
        engine=engine,
        max_pages=1,
        max_page_pixels=max_page_pixels,
        reading_order_resolver=reading_order_resolver,
        auto_orient_right_angles=auto_orient_right_angles,
        include_ocr_line_stats=(
            probe_right_angle_rotations and not auto_orient_right_angles
        ),
    )
    dimensions["ocr"] = _ocr_document_stats(document)
    if auto_orient_right_angles:
        pages_meta = document.metadata.get("auto_orientation", {}).get("pages", [])
        if pages_meta:
            dimensions["auto_orientation"] = pages_meta[0]
    dimensions["layer_gap"] = {
        "classification": "image-no-native-layer",
        "native_layer_empty": True,
        "native_layer_sparse": True,
        "ocr_minus_native_nonspace_characters": int(
            dimensions["ocr"]["text"]["nonspace_characters"]
        ),
        "ocr_minus_native_lao_characters": int(
            dimensions["ocr"]["text"]["lao_characters"]
        ),
        "ocr_to_native_nonspace_ratio": None,
    }
    dimensions["ocr_quality"] = _confidence_diagnostics(
        dimensions["ocr"]["mean_block_confidence"]
    )
    if probe_right_angle_rotations and not auto_orient_right_angles:
        dimensions["rotation_probe"] = _rotation_probe(
            path,
            baseline_ocr=dimensions["ocr"],
            engine=engine,
            max_page_pixels=max_page_pixels,
            reading_order_resolver=reading_order_resolver,
        )
    dimensions["elapsed_seconds"] = time.perf_counter() - started
    return {
        "page_count": 1,
        "selected_pages": [1],
        "layer_gap_summary": {"image-no-native-layer": 1},
        "pages": [dimensions],
    }


def evaluate_remote_sources(
    registry_path: str | Path,
    *,
    engine: OcrEngine,
    source_ids: Iterable[str] | None = None,
    all_remote: bool = False,
    requested_pages: Iterable[int] | None = None,
    max_pages_per_source: int = 3,
    max_source_bytes: int = 25 * 1024 * 1024,
    max_document_pages: int = 500,
    max_page_pixels: int = DEFAULT_MAX_PAGE_PIXELS,
    timeout_seconds: float = 20.0,
    reading_order_resolver: ReadingOrderResolver | None = None,
    probe_right_angle_rotations: bool = False,
    auto_orient_right_angles: bool = False,
    fetcher: RemoteFetcher = download_remote_source,
) -> dict[str, Any]:
    if max_document_pages < 1:
        raise ValueError("max_document_pages must be at least 1")
    if max_page_pixels < 1:
        raise ValueError("max_page_pixels must be at least 1")

    selected_pages = list(dict.fromkeys(requested_pages or []))
    sources = load_remote_registry_sources(
        registry_path,
        source_ids=source_ids,
        all_remote=all_remote,
    )

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="lao-ocr-remote-eval-") as temp:
        temp_root = Path(temp)
        for source in sources:
            source_id = str(source["id"])
            result: dict[str, Any] = {
                "id": source_id,
                "name": str(source.get("name", "")),
                "registry_status": str(source.get("status", "")),
                "registry_text_layer": (
                    source.get("evidence", {}).get("text_layer")
                    if isinstance(source.get("evidence"), dict)
                    else None
                ),
                "url": str(source.get("url", "")),
            }
            try:
                source_key = hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:24]
                source_dir = temp_root / source_key
                source_dir.mkdir(parents=True, exist_ok=True)
                downloaded = fetcher(
                    result["url"],
                    source_dir / "source",
                    max_bytes=max_source_bytes,
                    timeout_seconds=timeout_seconds,
                )
                result["download"] = {
                    key: value
                    for key, value in asdict(downloaded).items()
                    if key != "path"
                }
                result["download"]["path_persisted"] = False

                if downloaded.format == "pdf":
                    result["document"] = _evaluate_pdf(
                        downloaded.path,
                        engine=engine,
                        requested_pages=selected_pages,
                        max_pages_per_source=max_pages_per_source,
                        max_document_pages=max_document_pages,
                        max_page_pixels=max_page_pixels,
                        work_dir=source_dir,
                        reading_order_resolver=reading_order_resolver,
                        probe_right_angle_rotations=probe_right_angle_rotations,
                        auto_orient_right_angles=auto_orient_right_angles,
                    )
                else:
                    result["document"] = _evaluate_image(
                        downloaded.path,
                        engine=engine,
                        max_page_pixels=max_page_pixels,
                        reading_order_resolver=reading_order_resolver,
                        probe_right_angle_rotations=probe_right_angle_rotations,
                        auto_orient_right_angles=auto_orient_right_angles,
                    )
                result["status"] = "ok"
            except Exception as exc:
                result["status"] = "error"
                result["error"] = str(exc)
                result["error_type"] = exc.__class__.__name__
            results.append(result)

    ok_count = sum(1 for item in results if item["status"] == "ok")
    return {
        "schema_version": "1",
        "report_type": "remote-source-diagnostic",
        "not_benchmark_accuracy": True,
        "generated_at": datetime.now(UTC).isoformat(),
        "engine": engine.metadata(),
        "reading_order": (
            reading_order_resolver.metadata()
            if reading_order_resolver is not None
            else {"name": "DeterministicReadingOrderResolver"}
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "selection": {
            "source_ids": [str(source["id"]) for source in sources],
            "requested_pages": selected_pages,
            "max_pages_per_source": max_pages_per_source,
            "max_source_bytes": max_source_bytes,
            "max_document_pages": max_document_pages,
            "max_page_pixels": max_page_pixels,
            "probe_right_angle_rotations": probe_right_angle_rotations,
            "auto_orient_right_angles": auto_orient_right_angles,
        },
        "summary": {
            "sources": len(results),
            "ok": ok_count,
            "errors": len(results) - ok_count,
        },
        "sources": results,
    }


def write_remote_evaluation_report(report: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
