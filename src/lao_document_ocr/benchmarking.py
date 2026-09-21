from __future__ import annotations

import json
import platform
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from lao_document_ocr.dataset import DatasetSample, DatasetSplit, validate_dataset
from lao_document_ocr.metrics import _levenshtein
from lao_document_ocr.normalization import normalize_lao_text
from lao_document_ocr.ocr.base import OcrEngine
from lao_document_ocr.pipeline import process_document


@dataclass
class MetricCounts:
    character_edits: int = 0
    characters: int = 0
    word_edits: int = 0
    words: int = 0
    samples: int = 0

    @property
    def cer(self) -> float:
        if self.characters == 0:
            return 0.0 if self.character_edits == 0 else 1.0
        return self.character_edits / self.characters

    @property
    def wer(self) -> float:
        if self.words == 0:
            return 0.0 if self.word_edits == 0 else 1.0
        return self.word_edits / self.words

    def add(self, reference: str, hypothesis: str) -> None:
        ref_chars = list(reference)
        hyp_chars = list(hypothesis)
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        self.character_edits += _levenshtein(ref_chars, hyp_chars)
        self.characters += len(ref_chars)
        self.word_edits += _levenshtein(ref_words, hyp_words)
        self.words += len(ref_words)
        self.samples += 1

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "cer": self.cer,
            "wer": self.wer,
        }


def benchmark_dataset(
    samples: Iterable[DatasetSample],
    dataset_root: str | Path,
    engine: OcrEngine,
    *,
    split: DatasetSplit = DatasetSplit.TEST,
    verify_hashes: bool = True,
) -> dict:
    samples = list(samples)
    validation_errors = validate_dataset(samples, dataset_root, verify_hashes=verify_hashes)
    if validation_errors:
        joined = "\n".join(f"- {error}" for error in validation_errors)
        raise ValueError(f"Dataset validation failed:\n{joined}")

    selected = [sample for sample in samples if sample.split == split]
    if not selected:
        raise ValueError(f"No samples found for split '{split.value}'.")

    root = Path(dataset_root)
    overall = MetricCounts()
    subsets: dict[str, MetricCounts] = defaultdict(MetricCounts)
    tags: dict[str, MetricCounts] = defaultdict(MetricCounts)
    sample_results: list[dict] = []
    started = time.perf_counter()

    for sample in selected:
        source_path = root / sample.source
        truth_path = root / sample.ground_truth

        reference = normalize_lao_text(truth_path.read_text(encoding="utf-8"))
        item_started = time.perf_counter()
        document = process_document(
            source_path,
            source_name=source_path.name,
            engine=engine,
            max_pages=1,
        )
        hypothesis = normalize_lao_text(document.plain_text)
        elapsed = time.perf_counter() - item_started

        sample_counts = MetricCounts()
        sample_counts.add(reference, hypothesis)
        overall.add(reference, hypothesis)
        subsets[sample.subset.value].add(reference, hypothesis)
        for tag in sample.tags:
            tags[tag].add(reference, hypothesis)

        sample_results.append(
            {
                "id": sample.id,
                "document_id": sample.document_id,
                "subset": sample.subset.value,
                "tags": sample.tags,
                "cer": sample_counts.cer,
                "wer": sample_counts.wer,
                "elapsed_seconds": elapsed,
                "reference_characters": len(reference),
                "hypothesis_characters": len(hypothesis),
            }
        )

    elapsed_total = time.perf_counter() - started
    return {
        "schema_version": "1",
        "generated_at": datetime.now(UTC).isoformat(),
        "split": split.value,
        "engine": engine.metadata(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "elapsed_seconds": elapsed_total,
        "overall": overall.to_dict(),
        "subsets": {name: counts.to_dict() for name, counts in sorted(subsets.items())},
        "tags": {name: counts.to_dict() for name, counts in sorted(tags.items())},
        "samples": sample_results,
    }


def write_report(report: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    destination.write_text(payload, encoding="utf-8")
    return destination
