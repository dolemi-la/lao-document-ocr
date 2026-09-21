from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from pathlib import Path

from lao_document_ocr.capture_pack import (
    capture_pack_page,
    load_capture_pack,
)
from lao_document_ocr.dataset import DatasetSample, DatasetSubset
from lao_document_ocr.dataset_intake import add_dataset_sample

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class CaptureMode(StrEnum):
    FLATBED_SCAN = "flatbed-scan"
    DEGRADED_SCAN = "degraded-scan"
    PHONE_PHOTO = "phone-photo"


_CAPTURE_SUBSETS = {
    CaptureMode.FLATBED_SCAN: DatasetSubset.CLEAN_PRINT,
    CaptureMode.DEGRADED_SCAN: DatasetSubset.NOISY_SCAN,
    CaptureMode.PHONE_PHOTO: DatasetSubset.PHONE_PHOTO,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register_capture(
    *,
    capture_pack_manifest: str | Path,
    page_id: str,
    capture_image: str | Path,
    capture_id: str,
    capture_mode: CaptureMode,
    contributor: str,
    release_license: str,
    dataset_root: str | Path,
    dataset_manifest: str | Path,
    confirm_release: bool = False,
    notes: str | None = None,
) -> DatasetSample:
    if not confirm_release:
        raise ValueError(
            "Capture registration requires explicit confirmation that the contributor "
            "owns or controls the capture and releases it under the stated license."
        )
    if not capture_id.strip():
        raise ValueError("capture_id must not be empty")
    if not _SAFE_ID.fullmatch(capture_id):
        raise ValueError("capture_id may contain only letters, numbers, '.', '_' and '-'")
    if not contributor.strip():
        raise ValueError("contributor must not be empty")
    if not release_license.strip():
        raise ValueError("release_license must not be empty")

    pack_root, pack = load_capture_pack(capture_pack_manifest)
    page = capture_pack_page(pack, page_id)

    digital_page = pack_root / page.image
    ground_truth = pack_root / page.ground_truth
    if not digital_page.is_file():
        raise FileNotFoundError(f"Capture-pack page image not found: {digital_page}")
    if _sha256(digital_page) != page.sha256:
        raise ValueError(f"Capture-pack page image SHA-256 mismatch: {page_id}")
    if not ground_truth.is_file():
        raise FileNotFoundError(f"Capture-pack ground truth not found: {ground_truth}")

    capture_path = Path(capture_image)
    if not capture_path.is_file():
        raise FileNotFoundError(f"Capture image not found: {capture_path}")

    sample_id = f"{page_id}-{capture_id.strip()}"
    provenance = (
        f"Real-world {capture_mode.value} of project capture-pack page {page_id}; "
        f"capture contributed by {contributor.strip()}. "
        f"Source text: {pack.text_provenance} ({pack.text_license})."
    )
    combined_notes = (
        f"capture_pack={pack.pack_id}; page_id={page_id}; "
        f"text_license={pack.text_license}; text_provenance={pack.text_provenance}"
    )
    if notes:
        combined_notes += f"; {notes.strip()}"

    return add_dataset_sample(
        dataset_root=dataset_root,
        manifest_path=dataset_manifest,
        sample_id=sample_id,
        document_id=page_id,
        subset=_CAPTURE_SUBSETS[capture_mode],
        image_path=capture_path,
        ground_truth_path=ground_truth,
        license=release_license.strip(),
        provenance=provenance,
        notes=combined_notes,
        tags=[
            *page.tags,
            f"capture:{capture_mode.value}",
            "source:real-capture",
        ],
        rights_confirmed=True,
    )
