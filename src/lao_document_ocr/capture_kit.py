# ruff: noqa: E501
from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

from lao_document_ocr.capture_suite import load_capture_suite

_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_suite_file(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    if root not in path.parents and path != root:
        raise ValueError(f"Capture suite {label} escapes suite directory")
    if not path.is_file():
        raise FileNotFoundError(f"Capture suite {label} not found: {path}")
    return path


def _collector_worksheet(path: Path) -> bytes:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "combined_page",
            "template",
            "page_id",
            "required_capture_modes",
        }
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise ValueError(
                "Capture suite worksheet is missing required columns: "
                + ", ".join(missing)
            )
        rows = list(reader)

    output = io.StringIO(newline="")
    fieldnames = [
        "combined_page",
        "template",
        "page_id",
        "required_capture_modes",
        "capture_file",
        "notes",
    ]
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "combined_page": row["combined_page"],
                "template": row["template"],
                "page_id": row["page_id"],
                "required_capture_modes": row["required_capture_modes"],
                "capture_file": "",
                "notes": "",
            }
        )
    return output.getvalue().encode("utf-8")


def _instructions(
    *,
    suite_id: str,
    page_count: int,
    templates: list[str],
) -> bytes:
    template_lines = "\n".join(f"- {template}" for template in templates)
    text = f"""# Lao Document OCR capture campaign

Suite: {suite_id}
Printable pages: {page_count}

## Goal

Create real optical captures of the printed benchmark pages. Generated PDF/page images are not benchmark evidence by themselves.

## Included layouts

{template_lines}

## Capture guidance

1. Print {suite_id}.pdf at approximately 100% scale.
2. Keep the page-ID QR marker visible.
3. For flatbed scans, scan the complete page without digital cleanup.
4. For phone photos, use ordinary safe handheld capture with the full relevant page visible.
5. Do not submit screenshots, PDF exports, or simple re-encodes of the digital page.
6. Do not place unrelated private/confidential material in the frame.
7. Keep the original capture files. Avoid OCR, sharpening, beautification, or AI enhancement before submission.
8. Use capture-worksheet.csv to track page IDs and requested capture modes.

## File naming

QR page identification is preferred, so camera/scanner filenames do not need to be renamed. If you do rename files to page IDs, keep the printed QR visible; the registration tool rejects QR/filename disagreements.

## Submission

Send the original capture files to the benchmark maintainer together with the agreed contributor/release information.

The maintainer will run optical-evidence checks, bulk registration, manual review, readiness checks, and benchmark freeze before any public accuracy claim.

## Integrity

SHA256SUMS records the files in this kit. It can be used to confirm that the printable campaign was not altered in transit.
"""
    return text.encode("utf-8")


def _zip_write(
    archive: zipfile.ZipFile,
    name: str,
    data: bytes,
) -> None:
    info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    archive.writestr(info, data)


def build_capture_kit(
    suite_manifest: str | Path,
    output_zip: str | Path,
    *,
    source_revision: str | None = None,
) -> Path:
    suite_path = Path(suite_manifest).resolve()
    suite = load_capture_suite(suite_path)
    root = suite_path.parent.resolve()

    pdf_path = _safe_suite_file(
        root,
        suite.combined_pdf,
        "combined PDF",
    )
    if suite.worksheet is None:
        raise ValueError("Capture suite has no worksheet")
    worksheet_path = _safe_suite_file(
        root,
        suite.worksheet,
        "worksheet",
    )

    page_count = sum(pack.page_count for pack in suite.packs)
    templates = [pack.template for pack in suite.packs]

    pdf_name = f"{suite.suite_id}.pdf"
    worksheet_name = "capture-worksheet.csv"
    instructions_name = "CAPTURE-INSTRUCTIONS.md"
    kit_manifest_name = "capture-kit.json"
    checksums_name = "SHA256SUMS"

    payloads: dict[str, bytes] = {
        pdf_name: pdf_path.read_bytes(),
        worksheet_name: _collector_worksheet(worksheet_path),
        instructions_name: _instructions(
            suite_id=suite.suite_id,
            page_count=page_count,
            templates=templates,
        ),
    }

    kit_manifest = {
        "schema_version": "1",
        "suite_id": suite.suite_id,
        "source_revision": source_revision,
        "source_suite_manifest_sha256": _sha256_bytes(
            suite_path.read_bytes()
        ),
        "text_license": suite.text_license,
        "text_provenance": suite.text_provenance,
        "font": suite.font,
        "dpi": suite.dpi,
        "page_count": page_count,
        "templates": [
            {
                "template": pack.template,
                "page_count": pack.page_count,
            }
            for pack in suite.packs
        ],
        "files": {
            name: {
                "sha256": _sha256_bytes(data),
                "size_bytes": len(data),
            }
            for name, data in sorted(payloads.items())
        },
        "excludes_ground_truth": True,
        "excludes_digital_page_images": True,
        "excludes_internal_suite_paths": True,
    }
    payloads[kit_manifest_name] = (
        json.dumps(
            kit_manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    checksum_lines = [
        f"{_sha256_bytes(data)}  {name}"
        for name, data in sorted(payloads.items())
    ]
    payloads[checksums_name] = (
        "\n".join(checksum_lines) + "\n"
    ).encode("utf-8")

    destination = Path(output_zip)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(payloads):
            _zip_write(archive, name, payloads[name])

    return destination
