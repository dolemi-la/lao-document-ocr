from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from lao_document_ocr.capture_authenticity import (
    CaptureAuthenticity,
    compare_capture_to_digital_source,
)
from lao_document_ocr.capture_pack import load_capture_pack
from lao_document_ocr.capture_page_id import decode_page_id
from lao_document_ocr.capture_registration import (
    CaptureMode,
    register_capture,
)
from lao_document_ocr.capture_suite import load_capture_suite
from lao_document_ocr.dataset import (
    DatasetSubset,
    load_manifest,
)

_CAPTURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_MODE_SUBSETS = {
    CaptureMode.FLATBED_SCAN: DatasetSubset.CLEAN_PRINT,
    CaptureMode.DEGRADED_SCAN: DatasetSubset.NOISY_SCAN,
    CaptureMode.PHONE_PHOTO: DatasetSubset.PHONE_PHOTO,
}


@dataclass(frozen=True)
class CaptureBatchItem:
    page_id: str
    template: str
    pack_manifest: Path
    digital_page: Path
    capture_path: Path
    page_id_method: str
    authenticity: CaptureAuthenticity

    def to_dict(self) -> dict:
        return {
            "page_id": self.page_id,
            "template": self.template,
            "pack_manifest": str(self.pack_manifest),
            "digital_page": str(self.digital_page),
            "capture_path": str(self.capture_path),
            "page_id_method": self.page_id_method,
            "authenticity": self.authenticity.to_dict(),
        }


@dataclass(frozen=True)
class CaptureFileMapping:
    filename: str
    page_id: str
    template: str
    page_id_method: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CaptureBatchReport:
    suite_id: str
    capture_id: str
    capture_mode: str
    expected_pages: int
    discovered_captures: int
    planned_captures: int
    registered_captures: int
    missing_page_ids: tuple[str, ...]
    ignored_files: tuple[str, ...]
    registered_sample_ids: tuple[str, ...]
    dry_run: bool
    file_mappings: tuple[CaptureFileMapping, ...] = ()

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["missing_page_ids"] = list(self.missing_page_ids)
        payload["ignored_files"] = list(self.ignored_files)
        payload["registered_sample_ids"] = list(self.registered_sample_ids)
        payload["file_mappings"] = [
            mapping.to_dict() for mapping in self.file_mappings
        ]
        return payload


@dataclass(frozen=True)
class _SuitePage:
    page_id: str
    template: str
    pack_manifest: Path
    digital_page: Path
    requires_qr: bool


def _suite_pages(suite_manifest: Path) -> tuple[str, dict[str, _SuitePage]]:
    suite = load_capture_suite(suite_manifest)
    suite_root = suite_manifest.parent
    pages: dict[str, _SuitePage] = {}

    for pack in suite.packs:
        pack_manifest = (suite_root / pack.manifest).resolve()
        pack_root, manifest = load_capture_pack(pack_manifest)
        for page in manifest.pages:
            if page.id in pages:
                raise ValueError(
                    f"Duplicate page id across capture suite: {page.id}"
                )
            pages[page.id] = _SuitePage(
                page_id=page.id,
                template=pack.template,
                pack_manifest=pack_manifest,
                digital_page=(pack_root / page.image).resolve(),
                requires_qr="page-id:qr-v1" in page.tags,
            )

    if not pages:
        raise ValueError("Capture suite contains no pages")
    return suite.suite_id, pages


def _discover_capture_files(
    capture_dir: Path,
    suite_pages: dict[str, _SuitePage],
) -> tuple[dict[str, tuple[Path, str]], list[str]]:
    if not capture_dir.is_dir():
        raise FileNotFoundError(f"Capture directory not found: {capture_dir}")

    captures: dict[str, tuple[Path, str]] = {}
    ignored: list[str] = []
    for path in sorted(capture_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            ignored.append(path.name)
            continue
        if path.suffix.lower() not in _CAPTURE_EXTENSIONS:
            ignored.append(path.name)
            continue

        resolved = path.resolve()
        decoded = decode_page_id(resolved)
        filename_page_id = path.stem if path.stem in suite_pages else None

        if decoded is not None:
            if decoded not in suite_pages:
                raise ValueError(
                    f"{path.name}: QR page id is not part of the capture suite: "
                    f"{decoded}"
                )
            if filename_page_id is not None and filename_page_id != decoded:
                raise ValueError(
                    f"{path.name}: filename page id ({filename_page_id}) does not "
                    f"match QR page id ({decoded})"
                )
            page_id = decoded
            method = "qr"
        elif filename_page_id is not None:
            page = suite_pages[filename_page_id]
            if page.requires_qr:
                raise ValueError(
                    f"{path.name}: QR page id could not be decoded for a "
                    "page-id:qr-v1 capture suite"
                )
            page_id = filename_page_id
            method = "filename"
        else:
            raise ValueError(
                f"{path.name}: could not identify a capture-suite page from "
                "the QR marker or filename"
            )

        if page_id in captures:
            raise ValueError(
                f"Multiple capture files resolve to page id '{page_id}'"
            )
        captures[page_id] = (resolved, method)
    return captures, ignored


def _preflight_existing_samples(
    dataset_manifest: Path,
    sample_ids: set[str],
) -> None:
    if not dataset_manifest.is_file():
        return
    existing = load_manifest(dataset_manifest)
    collisions = sorted(
        sample.id
        for sample in existing
        if sample.id in sample_ids
    )
    if collisions:
        raise ValueError(
            "Capture sample ids already exist: " + ", ".join(collisions)
        )


def _destination_paths(
    *,
    dataset_root: Path,
    item: CaptureBatchItem,
    capture_id: str,
    capture_mode: CaptureMode,
) -> tuple[Path, Path]:
    subset = _MODE_SUBSETS[capture_mode].value
    sample_id = f"{item.page_id}-{capture_id}"
    image = (
        dataset_root
        / "data"
        / subset
        / f"{sample_id}{item.capture_path.suffix.lower()}"
    )
    truth = (
        dataset_root
        / "ground-truth"
        / subset
        / f"{sample_id}.txt"
    )
    return image, truth


def plan_capture_directory(
    *,
    suite_manifest: str | Path,
    capture_dir: str | Path,
    capture_id: str,
    capture_mode: CaptureMode,
    dataset_root: str | Path,
    dataset_manifest: str | Path,
    require_complete: bool = False,
) -> tuple[str, list[CaptureBatchItem], list[str], list[str]]:
    if not capture_id.strip():
        raise ValueError("capture_id must not be empty")
    if not _SAFE_ID.fullmatch(capture_id):
        raise ValueError(
            "capture_id may contain only letters, numbers, '.', '_' and '-'"
        )

    suite_path = Path(suite_manifest).resolve()
    suite_id, suite_pages = _suite_pages(suite_path)
    captures, ignored = _discover_capture_files(
        Path(capture_dir).resolve(),
        suite_pages,
    )

    missing = sorted(set(suite_pages) - set(captures))
    if require_complete and missing:
        raise ValueError(
            f"Capture directory is missing {len(missing)} suite pages: "
            + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )

    items: list[CaptureBatchItem] = []
    for page_id in sorted(captures):
        page = suite_pages[page_id]
        capture_path, page_id_method = captures[page_id]
        authenticity = compare_capture_to_digital_source(
            page.digital_page,
            capture_path,
        )
        if not authenticity.has_visible_content:
            raise ValueError(
                f"{page_id}: capture has too little visible page content"
            )
        if authenticity.likely_digital_copy:
            raise ValueError(
                f"{page_id}: capture is visually indistinguishable from "
                "the digital source page"
            )
        items.append(
            CaptureBatchItem(
                page_id=page_id,
                template=page.template,
                pack_manifest=page.pack_manifest,
                digital_page=page.digital_page,
                capture_path=capture_path,
                page_id_method=page_id_method,
                authenticity=authenticity,
            )
        )

    if not items:
        raise ValueError("Capture directory contains no suite capture images")

    sample_ids = {
        f"{item.page_id}-{capture_id}"
        for item in items
    }
    manifest_path = Path(dataset_manifest)
    _preflight_existing_samples(manifest_path, sample_ids)

    root = Path(dataset_root)
    for item in items:
        image_path, truth_path = _destination_paths(
            dataset_root=root,
            item=item,
            capture_id=capture_id,
            capture_mode=capture_mode,
        )
        if image_path.exists() or truth_path.exists():
            raise FileExistsError(
                f"Destination already exists for page {item.page_id}"
            )

    return suite_id, items, missing, ignored


def register_capture_directory(
    *,
    suite_manifest: str | Path,
    capture_dir: str | Path,
    capture_id: str,
    capture_mode: CaptureMode,
    contributor: str,
    release_license: str,
    dataset_root: str | Path,
    dataset_manifest: str | Path,
    confirm_release: bool = False,
    require_complete: bool = False,
    dry_run: bool = False,
    notes: str | None = None,
) -> CaptureBatchReport:
    if not contributor.strip():
        raise ValueError("contributor must not be empty")
    if not release_license.strip():
        raise ValueError("release_license must not be empty")
    if not dry_run and not confirm_release:
        raise ValueError(
            "Batch registration requires --confirm-release for real writes"
        )

    root = Path(dataset_root)
    manifest_path = Path(dataset_manifest)
    suite_id, items, missing, ignored = plan_capture_directory(
        suite_manifest=suite_manifest,
        capture_dir=capture_dir,
        capture_id=capture_id,
        capture_mode=capture_mode,
        dataset_root=root,
        dataset_manifest=manifest_path,
        require_complete=require_complete,
    )

    if dry_run:
        return CaptureBatchReport(
            suite_id=suite_id,
            capture_id=capture_id,
            capture_mode=capture_mode.value,
            expected_pages=len(items) + len(missing),
            discovered_captures=len(items),
            planned_captures=len(items),
            registered_captures=0,
            missing_page_ids=tuple(missing),
            ignored_files=tuple(ignored),
            registered_sample_ids=(),
            dry_run=True,
            file_mappings=tuple(
                CaptureFileMapping(
                    filename=item.capture_path.name,
                    page_id=item.page_id,
                    template=item.template,
                    page_id_method=item.page_id_method,
                )
                for item in items
            ),
        )

    manifest_existed = manifest_path.is_file()
    original_manifest = (
        manifest_path.read_bytes()
        if manifest_existed
        else None
    )
    registered_ids: list[str] = []

    try:
        for item in items:
            sample = register_capture(
                capture_pack_manifest=item.pack_manifest,
                page_id=item.page_id,
                capture_image=item.capture_path,
                capture_id=capture_id,
                capture_mode=capture_mode,
                contributor=contributor,
                release_license=release_license,
                dataset_root=root,
                dataset_manifest=manifest_path,
                confirm_release=True,
                notes=notes,
            )
            registered_ids.append(sample.id)
    except Exception:
        for item in items:
            image_path, truth_path = _destination_paths(
                dataset_root=root,
                item=item,
                capture_id=capture_id,
                capture_mode=capture_mode,
            )
            image_path.unlink(missing_ok=True)
            truth_path.unlink(missing_ok=True)
        if manifest_existed and original_manifest is not None:
            manifest_path.write_bytes(original_manifest)
        else:
            manifest_path.unlink(missing_ok=True)
        raise

    return CaptureBatchReport(
        suite_id=suite_id,
        capture_id=capture_id,
        capture_mode=capture_mode.value,
        expected_pages=len(items) + len(missing),
        discovered_captures=len(items),
        planned_captures=len(items),
        registered_captures=len(registered_ids),
        missing_page_ids=tuple(missing),
        ignored_files=tuple(ignored),
        registered_sample_ids=tuple(registered_ids),
        dry_run=False,
        file_mappings=tuple(
            CaptureFileMapping(
                filename=item.capture_path.name,
                page_id=item.page_id,
                template=item.template,
                page_id_method=item.page_id_method,
            )
            for item in items
        ),
    )


def write_capture_batch_report(
    report: CaptureBatchReport,
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
