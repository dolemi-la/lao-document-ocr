from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from lao_document_ocr.normalization import normalize_lao_text

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class CapturePackPage:
    id: str
    image: str
    ground_truth: str
    sha256: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class CapturePackManifest:
    schema_version: str
    pack_id: str
    text_license: str
    text_provenance: str
    font: str
    dpi: int
    pages: tuple[CapturePackPage, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["pages"] = [asdict(page) for page in self.pages]
        return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _graphemeish_clusters(text: str) -> list[str]:
    clusters: list[str] = []
    for char in text:
        if not clusters or not (
            unicodedata.combining(char) or unicodedata.category(char).startswith("M")
        ):
            clusters.append(char)
        else:
            clusters[-1] += char
    return clusters


def _fits(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> bool:
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return (right - left) <= width


def _hard_wrap(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    width: int,
) -> list[str]:
    clusters = _graphemeish_clusters(text)
    output: list[str] = []
    current: list[str] = []

    for cluster in clusters:
        candidate = "".join([*current, cluster])
        if current and not _fits(draw, candidate, font, width):
            output.append("".join(current))
            current = [cluster]
        else:
            current.append(cluster)

    if current:
        output.append("".join(current))
    return output


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    width: int,
) -> list[str]:
    text = normalize_lao_text(text)
    if not text:
        return []

    words = text.split(" ")
    if len(words) == 1:
        return _hard_wrap(draw, text, font, width)

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if _fits(draw, candidate, font, width):
            current = candidate
            continue

        if current:
            lines.append(current)
            current = ""

        if _fits(draw, word, font, width):
            current = word
        else:
            hard = _hard_wrap(draw, word, font, width)
            if hard:
                lines.extend(hard[:-1])
                current = hard[-1]

    if current:
        lines.append(current)
    return lines


def _render_page(
    *,
    page_id: str,
    lines: list[str],
    font_path: Path,
    dpi: int,
) -> tuple[Image.Image, str]:
    width = round(8.27 * dpi)
    height = round(11.69 * dpi)
    margin = round(0.75 * dpi)
    body_width = width - margin * 2

    body_font = ImageFont.truetype(str(font_path), size=max(18, round(dpi * 0.12)))
    meta_font = ImageFont.truetype(str(font_path), size=max(14, round(dpi * 0.075)))

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    y = margin
    draw.text(
        (margin, y),
        "Lao OCR Capture Pack",
        font=meta_font,
        fill="black",
    )
    y += round(dpi * 0.22)
    draw.text(
        (margin, y),
        f"Page ID: {page_id}",
        font=meta_font,
        fill="black",
    )
    y += round(dpi * 0.35)

    rendered_body: list[str] = []
    line_spacing = round(dpi * 0.21)
    paragraph_spacing = round(dpi * 0.12)
    bottom_limit = height - margin - round(dpi * 0.4)

    for source_line in lines:
        wrapped = wrap_text(draw, source_line, body_font, body_width)
        if not wrapped:
            continue

        for wrapped_line in wrapped:
            if y + line_spacing > bottom_limit:
                break
            draw.text((margin, y), wrapped_line, font=body_font, fill="black")
            rendered_body.append(wrapped_line)
            y += line_spacing
        y += paragraph_spacing
        if y + line_spacing > bottom_limit:
            break

    footer = f"Capture ID: {page_id}"
    draw.text(
        (margin, height - margin),
        footer,
        font=meta_font,
        fill="black",
        anchor="ls",
    )

    truth_lines = [
        "Lao OCR Capture Pack",
        f"Page ID: {page_id}",
        *rendered_body,
        footer,
    ]
    ground_truth = normalize_lao_text("\n".join(truth_lines)) + "\n"
    return image, ground_truth


def generate_capture_pack(
    corpus_lines: list[str],
    output_dir: str | Path,
    font_path: str | Path,
    *,
    pack_id: str,
    text_license: str,
    text_provenance: str,
    dpi: int = 150,
    lines_per_page: int = 10,
    max_pages: int | None = None,
) -> Path:
    if not corpus_lines:
        raise ValueError("corpus is empty")
    if not pack_id.strip():
        raise ValueError("pack_id must not be empty")
    if not _SAFE_ID.fullmatch(pack_id):
        raise ValueError("pack_id may contain only letters, numbers, '.', '_' and '-'")
    if not text_license.strip():
        raise ValueError("text_license must not be empty")
    if not text_provenance.strip():
        raise ValueError("text_provenance must not be empty")
    if dpi < 96:
        raise ValueError("dpi must be at least 96")
    if lines_per_page < 1:
        raise ValueError("lines_per_page must be at least 1")
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages must be at least 1")

    font = Path(font_path)
    if not font.is_file():
        raise FileNotFoundError(f"Font not found: {font}")

    normalized = [
        normalize_lao_text(line)
        for line in corpus_lines
        if normalize_lao_text(line)
    ]
    if not normalized:
        raise ValueError("corpus contains no usable lines")

    output = Path(output_dir)
    pages_dir = output / "pages"
    truth_dir = output / "ground-truth"
    pages_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    pages: list[CapturePackPage] = []
    page_images: list[Image.Image] = []

    for page_index, start in enumerate(range(0, len(normalized), lines_per_page), start=1):
        if max_pages is not None and page_index > max_pages:
            break

        page_id = f"{pack_id}-p{page_index:04d}"
        source_lines = normalized[start : start + lines_per_page]
        image, ground_truth = _render_page(
            page_id=page_id,
            lines=source_lines,
            font_path=font,
            dpi=dpi,
        )

        image_path = pages_dir / f"{page_id}.png"
        truth_path = truth_dir / f"{page_id}.txt"
        image.save(image_path, format="PNG", optimize=True)
        truth_path.write_text(ground_truth, encoding="utf-8")

        pages.append(
            CapturePackPage(
                id=page_id,
                image=image_path.relative_to(output).as_posix(),
                ground_truth=truth_path.relative_to(output).as_posix(),
                sha256=_sha256(image_path),
                lines=tuple(source_lines),
            )
        )
        page_images.append(image)

    if not pages:
        raise ValueError("capture pack generated no pages")

    pdf_path = output / f"{pack_id}.pdf"
    first, *rest = page_images
    first.save(
        pdf_path,
        "PDF",
        save_all=True,
        append_images=rest,
        resolution=float(dpi),
    )

    manifest = CapturePackManifest(
        schema_version="1",
        pack_id=pack_id,
        text_license=text_license,
        text_provenance=text_provenance,
        font=font.name,
        dpi=dpi,
        pages=tuple(pages),
    )
    manifest_path = output / "capture-pack.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _safe_relative_path(value: str, field: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Capture-pack {field} must stay inside the pack directory")
    return path.as_posix()


def load_capture_pack(path: str | Path) -> tuple[Path, CapturePackManifest]:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read capture pack: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid capture pack JSON: {exc}") from exc

    if payload.get("schema_version") != "1":
        raise ValueError("Unsupported capture pack schema version")

    page_entries = payload.get("pages")
    if not isinstance(page_entries, list) or not page_entries:
        raise ValueError("Capture pack contains no pages")

    pages = tuple(
        CapturePackPage(
            id=str(item["id"]),
            image=_safe_relative_path(str(item["image"]), "image"),
            ground_truth=_safe_relative_path(
                str(item["ground_truth"]),
                "ground_truth",
            ),
            sha256=str(item["sha256"]),
            lines=tuple(str(line) for line in item.get("lines", [])),
        )
        for item in page_entries
    )
    manifest = CapturePackManifest(
        schema_version="1",
        pack_id=str(payload["pack_id"]),
        text_license=str(payload["text_license"]),
        text_provenance=str(payload["text_provenance"]),
        font=str(payload["font"]),
        dpi=int(payload["dpi"]),
        pages=pages,
    )
    return manifest_path.parent, manifest


def capture_pack_page(
    manifest: CapturePackManifest,
    page_id: str,
) -> CapturePackPage:
    for page in manifest.pages:
        if page.id == page_id:
            return page
    raise ValueError(f"Capture pack page not found: {page_id}")
