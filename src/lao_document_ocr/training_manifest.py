from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from lao_document_ocr.normalization import normalize_lao_text

SPLIT_STRATEGY = "normalized-text-group-sha256-v1"


@dataclass(frozen=True)
class TrainingSample:
    id: str
    image: Path
    text: str
    sha256: str | None = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_training_manifest(
    manifest_path: str | Path,
    *,
    verify_hashes: bool = True,
) -> list[TrainingSample]:
    manifest = Path(manifest_path)
    root = manifest.parent
    samples: list[TrainingSample] = []
    ids: set[str] = set()

    with manifest.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc

            sample_id = payload.get("id")
            image = payload.get("image")
            text = payload.get("text")
            sha256 = payload.get("sha256")

            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"Manifest line {line_number}: id must be a non-empty string")
            if sample_id in ids:
                raise ValueError(f"Duplicate training sample id: {sample_id}")
            ids.add(sample_id)

            if not isinstance(image, str) or not image:
                raise ValueError(f"Manifest line {line_number}: image must be a string")
            if not isinstance(text, str) or not text:
                raise ValueError(f"Manifest line {line_number}: text must be a non-empty string")
            if sha256 is not None and not isinstance(sha256, str):
                raise ValueError(f"Manifest line {line_number}: sha256 must be a string")

            image_path = root / image
            if not image_path.is_file():
                raise ValueError(f"Training image does not exist: {image_path}")
            if verify_hashes and sha256 and _sha256(image_path) != sha256:
                raise ValueError(f"SHA-256 mismatch for sample {sample_id}")

            samples.append(
                TrainingSample(
                    id=sample_id,
                    image=image_path,
                    text=text,
                    sha256=sha256,
                )
            )

    if not samples:
        raise ValueError("Training manifest contains no samples")
    return samples


def _split_group_key(sample: TrainingSample) -> str:
    normalized = normalize_lao_text(sample.text)
    if not normalized:
        raise ValueError(f"Training sample {sample.id} has empty normalized text")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deterministic_split(
    samples: list[TrainingSample],
    *,
    dev_ratio: float = 0.1,
) -> tuple[list[TrainingSample], list[TrainingSample]]:
    if not 0 < dev_ratio < 1:
        raise ValueError("dev_ratio must be between 0 and 1")
    if len(samples) < 2:
        raise ValueError("At least two samples are required for a train/dev split")

    groups: dict[str, list[TrainingSample]] = {}
    for sample in samples:
        groups.setdefault(_split_group_key(sample), []).append(sample)

    if len(groups) < 2:
        raise ValueError(
            "At least two unique normalized text groups are required for a "
            "leakage-safe train/dev split"
        )

    threshold = int(dev_ratio * 10_000)
    dev_groups: set[str] = set()
    train_groups: set[str] = set()

    for group_key in sorted(groups):
        digest = bytes.fromhex(group_key)
        bucket = int.from_bytes(digest[:4], "big") % 10_000
        if bucket < threshold:
            dev_groups.add(group_key)
        else:
            train_groups.add(group_key)

    # Tiny datasets can hash entirely into one side. Move a whole text group,
    # never an individual augmented sample, so identical labels cannot leak.
    if not dev_groups:
        moved = sorted(train_groups)[-1]
        train_groups.remove(moved)
        dev_groups.add(moved)
    if not train_groups:
        moved = sorted(dev_groups)[0]
        dev_groups.remove(moved)
        train_groups.add(moved)

    train: list[TrainingSample] = []
    dev: list[TrainingSample] = []
    for sample in samples:
        group_key = _split_group_key(sample)
        if group_key in dev_groups:
            dev.append(sample)
        else:
            train.append(sample)

    return train, dev
