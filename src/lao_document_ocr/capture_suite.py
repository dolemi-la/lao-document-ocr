from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

from lao_document_ocr.capture_pack import generate_capture_pack, load_capture_pack
from lao_document_ocr.capture_templates import CaptureTemplate

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class CaptureSuitePack:
    template: str
    pack_id: str
    manifest: str
    printable_pdf: str
    page_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CaptureSuiteManifest:
    schema_version: str
    suite_id: str
    text_license: str
    text_provenance: str
    font: str
    dpi: int
    combined_pdf: str
    packs: tuple[CaptureSuitePack, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "text_license": self.text_license,
            "text_provenance": self.text_provenance,
            "font": self.font,
            "dpi": self.dpi,
            "combined_pdf": self.combined_pdf,
            "packs": [pack.to_dict() for pack in self.packs],
        }


def _merge_pdfs(pdf_paths: list[Path], destination: Path) -> None:
    output = pymupdf.open()
    try:
        for path in pdf_paths:
            source = pymupdf.open(path)
            try:
                output.insert_pdf(source)
            finally:
                source.close()
        output.save(destination)
    finally:
        output.close()


def generate_capture_suite(
    corpus_lines: list[str],
    output_dir: str | Path,
    font_path: str | Path,
    *,
    suite_id: str,
    text_license: str,
    text_provenance: str,
    templates: list[CaptureTemplate] | tuple[CaptureTemplate, ...] | None = None,
    dpi: int = 150,
    lines_per_page: int = 8,
    max_pages_per_template: int | None = None,
) -> Path:
    if not suite_id.strip():
        raise ValueError("suite_id must not be empty")
    if not _SAFE_ID.fullmatch(suite_id):
        raise ValueError(
            "suite_id may contain only letters, numbers, '.', '_' and '-'"
        )

    selected = tuple(templates or tuple(CaptureTemplate))
    if not selected:
        raise ValueError("At least one capture template is required")
    if len(set(selected)) != len(selected):
        raise ValueError("Capture suite templates must be unique")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    font = Path(font_path)
    packs: list[CaptureSuitePack] = []
    pack_pdfs: list[Path] = []

    for template in selected:
        pack_id = f"{suite_id}-{template.value}"
        pack_dir = output / template.value
        manifest_path = generate_capture_pack(
            corpus_lines,
            pack_dir,
            font,
            pack_id=pack_id,
            text_license=text_license,
            text_provenance=text_provenance,
            dpi=dpi,
            lines_per_page=lines_per_page,
            max_pages=max_pages_per_template,
            template=template,
        )
        _, manifest = load_capture_pack(manifest_path)
        pdf_path = pack_dir / f"{pack_id}.pdf"
        pack_pdfs.append(pdf_path)
        packs.append(
            CaptureSuitePack(
                template=template.value,
                pack_id=pack_id,
                manifest=manifest_path.relative_to(output).as_posix(),
                printable_pdf=pdf_path.relative_to(output).as_posix(),
                page_count=len(manifest.pages),
            )
        )

    combined_pdf = output / f"{suite_id}.pdf"
    _merge_pdfs(pack_pdfs, combined_pdf)

    suite = CaptureSuiteManifest(
        schema_version="1",
        suite_id=suite_id,
        text_license=text_license,
        text_provenance=text_provenance,
        font=font.name,
        dpi=dpi,
        combined_pdf=combined_pdf.name,
        packs=tuple(packs),
    )
    suite_manifest = output / "capture-suite.json"
    suite_manifest.write_text(
        json.dumps(suite.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return suite_manifest


def load_capture_suite(path: str | Path) -> CaptureSuiteManifest:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read capture suite: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid capture suite JSON: {exc}") from exc

    if payload.get("schema_version") != "1":
        raise ValueError("Unsupported capture suite schema version")

    raw_packs = payload.get("packs")
    if not isinstance(raw_packs, list) or not raw_packs:
        raise ValueError("Capture suite contains no packs")

    packs = tuple(
        CaptureSuitePack(
            template=str(item["template"]),
            pack_id=str(item["pack_id"]),
            manifest=str(item["manifest"]),
            printable_pdf=str(item["printable_pdf"]),
            page_count=int(item["page_count"]),
        )
        for item in raw_packs
    )
    return CaptureSuiteManifest(
        schema_version="1",
        suite_id=str(payload["suite_id"]),
        text_license=str(payload["text_license"]),
        text_provenance=str(payload["text_provenance"]),
        font=str(payload["font"]),
        dpi=int(payload["dpi"]),
        combined_pdf=str(payload["combined_pdf"]),
        packs=packs,
    )
