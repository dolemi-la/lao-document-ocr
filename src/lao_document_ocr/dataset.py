from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

_TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,63}$")


class DatasetSplit(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class DatasetSubset(StrEnum):
    CLEAN_PRINT = "clean-print"
    NOISY_SCAN = "noisy-scan"
    PHONE_PHOTO = "phone-photo"
    MIXED_LAO_ENGLISH = "mixed-lao-english"
    MULTI_COLUMN = "multi-column"
    SIMPLE_TABLE = "simple-table"
    COMPLEX_TABLE = "complex-table"
    RECEIPT = "receipt"
    FORM = "form"
    OTHER = "other"


class DatasetSample(BaseModel):
    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    document_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    split: DatasetSplit
    subset: DatasetSubset
    source: str = Field(min_length=1)
    ground_truth: str = Field(min_length=1)
    language: str = "lo"
    license: str = Field(min_length=1)
    provenance: str = Field(min_length=1)
    source_url: str | None = None
    license_url: str | None = None
    sha256: str | None = None
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: list[str]) -> list[str]:
        normalized: set[str] = set()
        for raw in value:
            tag = raw.strip().lower()
            if not tag:
                continue
            if not _TAG_PATTERN.fullmatch(tag):
                raise ValueError(
                    "tags may contain lowercase letters, numbers, '.', '_', ':' and '-'"
                )
            normalized.add(tag)
        return sorted(normalized)

    @field_validator("source", "ground_truth")
    @classmethod
    def relative_paths_only(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("dataset paths must be relative and stay inside the dataset root")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.lower()
        if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
            raise ValueError("sha256 must be a 64-character hexadecimal digest")
        return normalized


class DatasetManifestError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: str | Path) -> list[DatasetSample]:
    manifest_path = Path(path)
    samples: list[DatasetSample] = []
    seen_ids: set[str] = set()

    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetManifestError(f"Could not read manifest: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
            sample = DatasetSample.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise DatasetManifestError(
                f"Invalid manifest entry at line {line_number}: {exc}"
            ) from exc

        if sample.id in seen_ids:
            raise DatasetManifestError(f"Duplicate sample id: {sample.id}")
        seen_ids.add(sample.id)
        samples.append(sample)

    if not samples:
        raise DatasetManifestError("Manifest contains no samples.")
    return samples


def _split_leakage_errors(samples: Iterable[DatasetSample]) -> list[str]:
    document_splits: dict[str, set[DatasetSplit]] = {}
    for sample in samples:
        document_splits.setdefault(sample.document_id, set()).add(sample.split)

    errors: list[str] = []
    for document_id, splits in sorted(document_splits.items()):
        if len(splits) > 1:
            names = ", ".join(sorted(split.value for split in splits))
            errors.append(
                f"document {document_id} appears in multiple splits: {names}"
            )
    return errors


def validate_dataset(
    samples: Iterable[DatasetSample],
    dataset_root: str | Path,
    *,
    verify_hashes: bool = True,
) -> list[str]:
    samples = list(samples)
    root = Path(dataset_root).resolve()
    errors = _split_leakage_errors(samples)

    observed_hashes: dict[str, list[str]] = {}

    for sample in samples:
        source = (root / sample.source).resolve()
        truth = (root / sample.ground_truth).resolve()

        try:
            source.relative_to(root)
            truth.relative_to(root)
        except ValueError:
            errors.append(f"{sample.id}: path escapes dataset root")
            continue

        if not source.is_file():
            errors.append(f"{sample.id}: missing source file {sample.source}")
        if not truth.is_file():
            errors.append(f"{sample.id}: missing ground truth file {sample.ground_truth}")

        observed_hash = sample.sha256
        if verify_hashes and source.is_file():
            actual = sha256_file(source)
            if sample.sha256 and actual != sample.sha256:
                errors.append(
                    f"{sample.id}: sha256 mismatch (expected {sample.sha256}, got {actual})"
                )
            observed_hash = actual

        if observed_hash:
            observed_hashes.setdefault(observed_hash, []).append(sample.id)

        if truth.is_file():
            try:
                text = truth.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                errors.append(f"{sample.id}: ground truth must be UTF-8")
            else:
                if not text.strip():
                    errors.append(f"{sample.id}: ground truth is empty")

    for digest, sample_ids in sorted(observed_hashes.items()):
        unique_ids = sorted(set(sample_ids))
        if len(unique_ids) > 1:
            errors.append(
                "duplicate source image sha256 "
                f"{digest}: {', '.join(unique_ids)}"
            )

    return errors
