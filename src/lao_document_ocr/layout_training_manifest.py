from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from lao_document_ocr.dataset import (
    DatasetSample,
    DatasetSplit,
    validate_dataset,
)
from lao_document_ocr.layout_ground_truth import load_layout_ground_truth


@dataclass(frozen=True)
class LayoutTrainingEntry:
    id: str
    document_id: str
    image: str
    layout_ground_truth: str
    split: str
    subset: str
    tags: tuple[str, ...]
    sha256: str | None
    width: int
    height: int
    block_counts: dict[str, int]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["tags"] = list(self.tags)
        return payload


def build_layout_training_entries(
    samples: list[DatasetSample],
    dataset_root: str | Path,
    *,
    splits: set[DatasetSplit] | None = None,
) -> list[LayoutTrainingEntry]:
    if not samples:
        raise ValueError("Layout training manifest requires dataset samples")

    root = Path(dataset_root).resolve()
    errors = validate_dataset(samples, root)
    if errors:
        raise ValueError(
            "Dataset validation failed before layout manifest generation: "
            + "; ".join(errors)
        )

    selected = [
        sample
        for sample in samples
        if sample.layout_ground_truth is not None
        and (splits is None or sample.split in splits)
    ]
    if not selected:
        raise ValueError(
            "No layout-labeled samples match the requested split selection"
        )

    entries: list[LayoutTrainingEntry] = []
    for sample in sorted(selected, key=lambda item: item.id):
        layout_path = root / sample.layout_ground_truth
        document = load_layout_ground_truth(layout_path)
        page = document.pages[0]
        counts = Counter(block.type.value for block in page.blocks)

        entries.append(
            LayoutTrainingEntry(
                id=sample.id,
                document_id=sample.document_id,
                image=sample.source,
                layout_ground_truth=sample.layout_ground_truth,
                split=sample.split.value,
                subset=sample.subset.value,
                tags=tuple(sample.tags),
                sha256=sample.sha256,
                width=page.width,
                height=page.height,
                block_counts=dict(sorted(counts.items())),
            )
        )

    return entries


def write_layout_training_manifest(
    entries: list[LayoutTrainingEntry],
    path: str | Path,
) -> Path:
    if not entries:
        raise ValueError("Cannot write an empty layout training manifest")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(
                json.dumps(
                    entry.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return destination


def summarize_layout_training_entries(
    entries: list[LayoutTrainingEntry],
) -> dict:
    if not entries:
        raise ValueError("Cannot summarize an empty layout training manifest")

    by_split = Counter(entry.split for entry in entries)
    by_subset = Counter(entry.subset for entry in entries)
    by_tag = Counter(tag for entry in entries for tag in entry.tags)
    by_block_type: Counter[str] = Counter()

    for entry in entries:
        by_block_type.update(entry.block_counts)

    return {
        "samples": len(entries),
        "by_split": dict(sorted(by_split.items())),
        "by_subset": dict(sorted(by_subset.items())),
        "by_tag": dict(sorted(by_tag.items())),
        "by_block_type": dict(sorted(by_block_type.items())),
    }


def _safe_relative_manifest_path(value: str, field: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"layout training {field} must be a safe relative path"
        )
    return path.as_posix()


def load_layout_training_manifest(
    path: str | Path,
) -> list[LayoutTrainingEntry]:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(
            f"Could not read layout training manifest: {exc}"
        ) from exc

    entries: list[LayoutTrainingEntry] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid layout training manifest JSON at line {line_number}: {exc}"
            ) from exc

        try:
            sample_id = str(payload["id"])
            document_id = str(payload["document_id"])
            image = _safe_relative_manifest_path(
                str(payload["image"]),
                "image",
            )
            layout_ground_truth = _safe_relative_manifest_path(
                str(payload["layout_ground_truth"]),
                "layout_ground_truth",
            )
            split = str(payload["split"])
            subset = str(payload["subset"])
            width = int(payload["width"])
            height = int(payload["height"])
            tags = tuple(str(tag) for tag in payload.get("tags", []))
            block_counts = {
                str(name): int(count)
                for name, count in dict(payload.get("block_counts", {})).items()
            }
            sha256 = payload.get("sha256")
            if sha256 is not None:
                sha256 = str(sha256)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid layout training manifest entry at line {line_number}: {exc}"
            ) from exc

        if not sample_id or not document_id:
            raise ValueError(
                f"Invalid layout training manifest entry at line {line_number}: "
                "id/document_id must not be empty"
            )
        if width < 1 or height < 1:
            raise ValueError(
                f"Invalid layout training manifest entry at line {line_number}: "
                "width/height must be positive"
            )
        if any(count < 0 for count in block_counts.values()):
            raise ValueError(
                f"Invalid layout training manifest entry at line {line_number}: "
                "block counts must be non-negative"
            )
        if sample_id in seen_ids:
            raise ValueError(
                f"Duplicate layout training sample id: {sample_id}"
            )
        seen_ids.add(sample_id)

        entries.append(
            LayoutTrainingEntry(
                id=sample_id,
                document_id=document_id,
                image=image,
                layout_ground_truth=layout_ground_truth,
                split=split,
                subset=subset,
                tags=tags,
                sha256=sha256,
                width=width,
                height=height,
                block_counts=block_counts,
            )
        )

    if not entries:
        raise ValueError("Layout training manifest contains no samples")
    return entries
