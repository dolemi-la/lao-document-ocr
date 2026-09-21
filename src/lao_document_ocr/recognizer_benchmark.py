from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from lao_document_ocr.metrics import _levenshtein
from lao_document_ocr.normalization import normalize_lao_text
from lao_document_ocr.training_manifest import TrainingSample


class LineRecognizer(Protocol):
    def recognize(self, image_path: str | Path): ...


@dataclass
class RecognitionMetricCounts:
    character_edits: int = 0
    characters: int = 0
    word_edits: int = 0
    words: int = 0
    samples: int = 0

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
        payload = asdict(self)
        payload["cer"] = (
            self.character_edits / self.characters
            if self.characters
            else (0.0 if self.character_edits == 0 else 1.0)
        )
        payload["wer"] = (
            self.word_edits / self.words
            if self.words
            else (0.0 if self.word_edits == 0 else 1.0)
        )
        return payload


def benchmark_recognizer(
    samples: list[TrainingSample],
    recognizer: LineRecognizer,
) -> dict:
    if not samples:
        raise ValueError("No recognizer benchmark samples supplied")

    overall = RecognitionMetricCounts()
    sample_results: list[dict] = []
    started = time.perf_counter()

    for sample in samples:
        reference = normalize_lao_text(sample.text)
        item_started = time.perf_counter()
        result = recognizer.recognize(sample.image)
        elapsed = time.perf_counter() - item_started
        hypothesis = normalize_lao_text(result.text)

        counts = RecognitionMetricCounts()
        counts.add(reference, hypothesis)
        overall.add(reference, hypothesis)

        sample_results.append(
            {
                "id": sample.id,
                "reference": reference,
                "hypothesis": hypothesis,
                "cer": counts.to_dict()["cer"],
                "wer": counts.to_dict()["wer"],
                "elapsed_seconds": elapsed,
                "uncalibrated_confidence": getattr(result, "confidence", None),
                "calibrated_confidence": getattr(result, "calibrated_confidence", None),
            }
        )

    return {
        "schema_version": "1",
        "model": getattr(recognizer, "metadata", None),
        "elapsed_seconds": time.perf_counter() - started,
        "overall": overall.to_dict(),
        "samples": sample_results,
    }


def write_recognizer_report(report: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
