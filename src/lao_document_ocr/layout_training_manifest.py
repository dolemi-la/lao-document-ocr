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
