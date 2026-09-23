from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lao_document_ocr.normalization import normalize_lao_text
from lao_document_ocr.vocabulary import CharacterVocabulary

_BOS_ID = -1


def _context_key(context: tuple[int, ...]) -> str:
    return ",".join(str(token) for token in context)


def _parse_context_key(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(item) for item in value.split(","))


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LanguageModelTrainingStats:
    total_lines: int
    used_lines: int
    skipped_lines: int
    tokens: int

    def to_dict(self) -> dict[str, int]:
        return {
            "total_lines": self.total_lines,
            "used_lines": self.used_lines,
            "skipped_lines": self.skipped_lines,
            "tokens": self.tokens,
        }


@dataclass(frozen=True)
class CharacterNgramLanguageModel:
    order: int
    alpha: float
    vocabulary_checksum: str
    vocabulary_size: int
    counts: dict[tuple[int, ...], dict[int, int]]
    totals: dict[tuple[int, ...], int]
    artifact_sha256: str | None = None
    training_stats: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("language model order must be at least 1")
        if self.alpha <= 0:
            raise ValueError("language model alpha must be positive")
        if self.vocabulary_size < 1:
            raise ValueError("language model vocabulary_size must be positive")

    def _context(self, prefix: tuple[int, ...], length: int) -> tuple[int, ...]:
        if length == 0:
            return ()
        needed = max(0, length - len(prefix))
        padded = (_BOS_ID,) * needed + prefix
        return tuple(padded[-length:])

    def score_extension(self, prefix: tuple[int, ...], token_id: int) -> float:
        if not 1 <= token_id <= self.vocabulary_size:
            raise ValueError(f"token id {token_id} is outside language-model vocabulary")

        max_context = min(self.order - 1, max(self.order - 1, len(prefix)))
        for length in range(max_context, -1, -1):
            context = self._context(prefix, length)
            next_counts = self.counts.get(context)
            total = self.totals.get(context)
            if next_counts is None or total is None:
                continue
            count = next_counts.get(token_id, 0)
            probability = (count + self.alpha) / (
                total + self.alpha * self.vocabulary_size
            )
            return math.log(probability)

        return -math.log(self.vocabulary_size)

    def score_sequence(self, token_ids: tuple[int, ...]) -> float:
        score = 0.0
        prefix: tuple[int, ...] = ()
        for token_id in token_ids:
            score += self.score_extension(prefix, token_id)
            prefix = (*prefix, token_id)
        return score

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "character-ngram",
            "order": self.order,
            "alpha": self.alpha,
            "vocabulary_checksum": self.vocabulary_checksum,
            "vocabulary_size": self.vocabulary_size,
            "sha256": self.artifact_sha256,
            "training_stats": self.training_stats,
        }

    @classmethod
    def load(cls, path: str | Path) -> CharacterNgramLanguageModel:
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "1":
            raise ValueError("Unsupported language model schema version")
        if payload.get("type") != "character-ngram":
            raise ValueError("Unsupported language model type")

        raw_contexts = payload.get("contexts")
        if not isinstance(raw_contexts, dict):
            raise ValueError("Language model contexts must be an object")

        counts: dict[tuple[int, ...], dict[int, int]] = {}
        totals: dict[tuple[int, ...], int] = {}
        for raw_context, item in raw_contexts.items():
            if not isinstance(item, dict):
                raise ValueError("Language model context entry must be an object")
            context = _parse_context_key(str(raw_context))
            raw_next = item.get("next")
            total = item.get("total")
            if not isinstance(raw_next, dict) or not isinstance(total, int):
                raise ValueError("Invalid language model context counts")
            parsed_next = {int(token): int(count) for token, count in raw_next.items()}
            vocabulary_size = int(payload["vocabulary_size"])
            if any(
                token < 1 or token > vocabulary_size
                for token in parsed_next
            ):
                raise ValueError("Language model next-token id is outside vocabulary")
            if any(
                token != _BOS_ID and (token < 1 or token > vocabulary_size)
                for token in context
            ):
                raise ValueError("Language model context token is outside vocabulary")
            if len(context) > int(payload["order"]) - 1:
                raise ValueError("Language model context exceeds configured order")
            if any(count < 1 for count in parsed_next.values()) or total < 1:
                raise ValueError("Language model counts must be positive")
            if sum(parsed_next.values()) != total:
                raise ValueError("Language model context total does not match counts")
            counts[context] = parsed_next
            totals[context] = total

        return cls(
            order=int(payload["order"]),
            alpha=float(payload["alpha"]),
            vocabulary_checksum=str(payload["vocabulary_checksum"]),
            vocabulary_size=int(payload["vocabulary_size"]),
            counts=counts,
            totals=totals,
            artifact_sha256=_sha256_file(source),
            training_stats=(
                dict(payload["training_stats"])
                if isinstance(payload.get("training_stats"), dict)
                else None
            ),
        )


def train_character_ngram_language_model(
    lines: list[str],
    vocabulary: CharacterVocabulary,
    *,
    order: int = 3,
    alpha: float = 0.1,
) -> tuple[CharacterNgramLanguageModel, LanguageModelTrainingStats]:
    if order < 1 or order > 6:
        raise ValueError("language model order must be in [1, 6]")
    if alpha <= 0:
        raise ValueError("language model alpha must be positive")

    context_counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    total_lines = len(lines)
    used_lines = 0
    skipped_lines = 0
    token_count = 0

    for raw in lines:
        text = normalize_lao_text(raw)
        if not text:
            continue
        try:
            tokens = vocabulary.encode(text)
        except ValueError:
            skipped_lines += 1
            continue
        if not tokens:
            continue

        used_lines += 1
        token_count += len(tokens)
        history: tuple[int, ...] = ()
        for token_id in tokens:
            max_context = order - 1
            for length in range(0, max_context + 1):
                needed = max(0, length - len(history))
                padded = (_BOS_ID,) * needed + history
                context = tuple(padded[-length:]) if length else ()
                context_counts[context][token_id] += 1
            history = (*history, token_id)

    if used_lines == 0:
        raise ValueError("No corpus lines could be encoded by the vocabulary")

    counts = {
        context: dict(counter)
        for context, counter in context_counts.items()
    }
    totals = {
        context: sum(counter.values())
        for context, counter in context_counts.items()
    }
    stats = LanguageModelTrainingStats(
        total_lines=total_lines,
        used_lines=used_lines,
        skipped_lines=skipped_lines,
        tokens=token_count,
    )
    return (
        CharacterNgramLanguageModel(
            order=order,
            alpha=alpha,
            vocabulary_checksum=vocabulary.checksum(),
            vocabulary_size=vocabulary.size - 1,
            counts=counts,
            totals=totals,
            training_stats=stats.to_dict(),
        ),
        stats,
    )


def save_language_model(
    model: CharacterNgramLanguageModel,
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    contexts = {
        _context_key(context): {
            "total": model.totals[context],
            "next": {
                str(token): count
                for token, count in sorted(model.counts[context].items())
            },
        }
        for context in sorted(model.counts)
    }
    payload = {
        "schema_version": "1",
        "type": "character-ngram",
        "order": model.order,
        "alpha": model.alpha,
        "vocabulary_checksum": model.vocabulary_checksum,
        "vocabulary_size": model.vocabulary_size,
        "training_stats": model.training_stats,
        "contexts": contexts,
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
