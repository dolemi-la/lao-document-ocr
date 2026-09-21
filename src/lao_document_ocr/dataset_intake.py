from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from lao_document_ocr.dataset import (
    DatasetSample,
    DatasetSplit,
    DatasetSubset,
    load_manifest,
    sha256_file,
)


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.8
    dev: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        values = (self.train, self.dev, self.test)
        if any(value < 0 for value in values):
            raise ValueError("split ratios must be non-negative")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1")
        if self.train <= 0 or self.test <= 0:
            raise ValueError("train and test ratios must be greater than 0")


def stable_document_split(
    document_id: str,
    ratios: SplitRatios | None = None,
) -> DatasetSplit:
    ratios = ratios or SplitRatios()
    digest = hashlib.sha256(document_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / (2**64)

    if bucket < ratios.train:
        return DatasetSplit.TRAIN
    if bucket < ratios.train + ratios.dev:
        return DatasetSplit.DEV
    return DatasetSplit.TEST


def _manifest_samples_if_exists(path: Path) -> list[DatasetSample]:
    if not path.exists():
        return []
    return load_manifest(path)


def _existing_document_split(
    samples: list[DatasetSample],
    document_id: str,
) -> DatasetSplit | None:
    splits = {sample.split for sample in samples if sample.document_id == document_id}
    if not splits:
        return None
    if len(splits) != 1:
        raise ValueError(f"Existing manifest already has split leakage for {document_id}")
    return next(iter(splits))


def _safe_extension(path: Path) -> str:
    suffix = path.suffix.lower()
    allowed = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
    if suffix not in allowed:
        raise ValueError(f"Unsupported benchmark image type: {suffix or 'unknown'}")
    return suffix


def add_dataset_sample(
    *,
    dataset_root: str | Path,
    manifest_path: str | Path,
    sample_id: str,
    document_id: str,
    subset: DatasetSubset,
    image_path: str | Path,
    ground_truth_path: str | Path,
    license: str,
    provenance: str,
    language: str = "lo",
    source_url: str | None = None,
    license_url: str | None = None,
    notes: str | None = None,
    tags: list[str] | tuple[str, ...] | None = None,
    split: DatasetSplit | None = None,
    rights_confirmed: bool = False,
) -> DatasetSample:
    if not rights_confirmed:
        raise ValueError(
            "Dataset intake requires explicit confirmation that the sample may be "
            "redistributed and used for OCR/model evaluation."
        )
    if not license.strip():
        raise ValueError("license must not be empty")
    if not provenance.strip():
        raise ValueError("provenance must not be empty")

    source_image = Path(image_path)
    source_truth = Path(ground_truth_path)
    if not source_image.is_file():
        raise FileNotFoundError(f"Source image not found: {source_image}")
    if not source_truth.is_file():
        raise FileNotFoundError(f"Ground truth not found: {source_truth}")

    try:
        truth_text = source_truth.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Ground truth must be UTF-8") from exc
    if not truth_text.strip():
        raise ValueError("Ground truth must not be empty")

    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    existing = _manifest_samples_if_exists(manifest)
    if any(sample.id == sample_id for sample in existing):
        raise ValueError(f"Sample id already exists: {sample_id}")

    existing_split = _existing_document_split(existing, document_id)
    chosen_split = existing_split or split or stable_document_split(document_id)
    if split is not None and existing_split is not None and split != existing_split:
        raise ValueError(
            f"Document {document_id} is already assigned to split {existing_split.value}"
        )

    suffix = _safe_extension(source_image)
    image_relative = Path("data") / subset.value / f"{sample_id}{suffix}"
    truth_relative = Path("ground-truth") / subset.value / f"{sample_id}.txt"
    destination_image = root / image_relative
    destination_truth = root / truth_relative

    if destination_image.exists() or destination_truth.exists():
        raise FileExistsError(f"Destination for sample {sample_id} already exists")

    destination_image.parent.mkdir(parents=True, exist_ok=True)
    destination_truth.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(source_image, destination_image)
        destination_truth.write_text(truth_text, encoding="utf-8")
    except Exception:
        destination_image.unlink(missing_ok=True)
        destination_truth.unlink(missing_ok=True)
        raise

    sample = DatasetSample(
        id=sample_id,
        document_id=document_id,
        split=chosen_split,
        subset=subset,
        source=image_relative.as_posix(),
        ground_truth=truth_relative.as_posix(),
        language=language,
        license=license.strip(),
        provenance=provenance.strip(),
        source_url=source_url,
        license_url=license_url,
        sha256=sha256_file(destination_image),
        notes=notes,
        tags=list(tags or []),
    )

    payload = json.dumps(sample.model_dump(mode="json", exclude_none=True), ensure_ascii=False)
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")

    return sample
