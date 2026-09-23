from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from lao_document_ocr.capture_batch import (
    CaptureBatchReport,
    register_capture_directory,
)
from lao_document_ocr.capture_pack import load_capture_pack
from lao_document_ocr.capture_registration import CaptureMode
from lao_document_ocr.capture_submission import (
    CaptureSubmissionError,
    extract_capture_submission,
    verify_capture_submission,
)
from lao_document_ocr.capture_suite import load_capture_suite

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CaptureSubmissionImportReport:
    suite_id: str
    source_revision: str | None
    kit_sha256: str
    capture_id: str
    capture_mode: str
    captured_pages: int
    expected_pages: int
    submission_complete: bool
    batch: CaptureBatchReport

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["batch"] = self.batch.to_dict()
        return payload


def _normalize_expected_hash(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not _HEX64.fullmatch(normalized):
        raise ValueError("expected_kit_sha256 must be a 64-character SHA-256 hex digest")
    return normalized


def register_capture_submission(
    *,
    submission_path: str | Path,
    suite_manifest: str | Path,
    contributor: str,
    release_license: str,
    dataset_root: str | Path,
    dataset_manifest: str | Path,
    confirm_release: bool = False,
    require_complete: bool = False,
    dry_run: bool = False,
    notes: str | None = None,
    expected_kit_sha256: str | None = None,
    expected_source_revision: str | None = None,
    max_page_pixels: int = 80_000_000,
) -> CaptureSubmissionImportReport:
    source = Path(submission_path).resolve()
    suite_path = Path(suite_manifest).resolve()
    suite = load_capture_suite(suite_path)
    suite_page_count = 0
    for pack in suite.packs:
        _, pack_manifest = load_capture_pack(
            (suite_path.parent / pack.manifest).resolve()
        )
        suite_page_count += len(pack_manifest.pages)

    with tempfile.TemporaryDirectory(prefix="lao-ocr-capture-submission-") as temp:
        temp_root = Path(temp)
        staged_submission = temp_root / "submission.zip"
        shutil.copyfile(source, staged_submission)
        try:
            staged_submission.chmod(0o600)
        except OSError:
            pass

        submission = verify_capture_submission(
            staged_submission,
            max_page_pixels=max_page_pixels,
        )

        if submission.suite_id != suite.suite_id:
            raise CaptureSubmissionError(
                "Capture submission suite does not match suite manifest: "
                f"{submission.suite_id} != {suite.suite_id}"
            )
        if submission.expected_pages != suite_page_count:
            raise CaptureSubmissionError(
                "Capture submission expected page count does not match suite manifest: "
                f"{submission.expected_pages} != {suite_page_count}"
            )
        if require_complete and not submission.complete:
            raise CaptureSubmissionError(
                "Capture submission is incomplete "
                f"({submission.captured_pages}/{submission.expected_pages})."
            )

        expected_hash = _normalize_expected_hash(expected_kit_sha256)
        if expected_hash is not None and submission.kit_sha256 != expected_hash:
            raise CaptureSubmissionError(
                "Capture submission kit SHA-256 does not match the expected kit."
            )

        if (
            expected_source_revision is not None
            and submission.source_revision != expected_source_revision
        ):
            raise CaptureSubmissionError(
                "Capture submission source revision does not match the expected revision."
            )

        capture_dir = extract_capture_submission(
            staged_submission,
            temp_root / "captures",
            max_page_pixels=max_page_pixels,
        )
        (capture_dir / "capture-submission.json").unlink(missing_ok=True)
        batch = register_capture_directory(
            suite_manifest=suite_path,
            capture_dir=capture_dir,
            capture_id=submission.capture_id,
            capture_mode=CaptureMode(submission.mode),
            contributor=contributor,
            release_license=release_license,
            dataset_root=dataset_root,
            dataset_manifest=dataset_manifest,
            confirm_release=confirm_release,
            require_complete=require_complete,
            dry_run=dry_run,
            notes=notes,
        )

    return CaptureSubmissionImportReport(
        suite_id=submission.suite_id,
        source_revision=submission.source_revision,
        kit_sha256=submission.kit_sha256,
        capture_id=submission.capture_id,
        capture_mode=submission.mode,
        captured_pages=submission.captured_pages,
        expected_pages=submission.expected_pages,
        submission_complete=submission.complete,
        batch=batch,
    )
