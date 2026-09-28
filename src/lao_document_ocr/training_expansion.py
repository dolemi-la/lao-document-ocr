"""Audit matched training-only expansion without publishing labels or image paths."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from lao_document_ocr.normalization import normalize_lao_text
from lao_document_ocr.training_manifest import (
    SPLIT_STRATEGY,
    TrainingSample,
    deterministic_split,
)
from lao_document_ocr.vocabulary import CharacterVocabulary


def _verified_identities(samples: list[TrainingSample]) -> dict[str, tuple[str, str]]:
    identities: dict[str, tuple[str, str]] = {}
    for sample in samples:
        if not isinstance(sample.id, str) or not sample.id:
            raise ValueError("Training sample ID must be a non-empty string")
        if sample.id in identities:
            raise ValueError("Duplicate training sample ID")
        if not isinstance(sample.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sample.sha256):
            raise ValueError("Matched expansion requires a pinned SHA-256 for every image")
        text = normalize_lao_text(sample.text)
        if not text:
            raise ValueError("Training sample has empty normalized text")
        digest = hashlib.sha256()
        with Path(sample.image).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != sample.sha256:
            raise ValueError("Training expansion image SHA-256 mismatch")
        identities[sample.id] = (text, sample.sha256)
    return identities


def _fingerprint(
    samples: list[TrainingSample],
    identities: dict[str, tuple[str, str]],
) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        text, image_sha256 = identities[sample.id]
        record = json.dumps(
            [sample.id, text, image_sha256],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(record).to_bytes(8, "big"))
        digest.update(record)
    return digest.hexdigest()


def audit_training_expansion(
    baseline: list[TrainingSample],
    expanded: list[TrainingSample],
    *,
    dev_ratio: float = 0.1,
) -> dict[str, Any]:
    """Require identical ordered dev, retained baseline samples, and fixed vocabulary.

    This audit verifies normalized exact labels and pinned image bytes, not source
    rights, document-level separation, near duplicates, or comparable optimization.
    It does not generate samples or alter the deterministic split algorithm.
    """
    base_ids = _verified_identities(baseline)
    expanded_ids = _verified_identities(expanded)
    if not base_ids.keys() <= expanded_ids.keys():
        raise ValueError("Expanded manifest must retain every original sample")
    if any(expanded_ids[key] != value for key, value in base_ids.items()):
        raise ValueError("Expanded manifest changed an original image or normalized label")

    base_train, base_dev = deterministic_split(baseline, dev_ratio=dev_ratio)
    new_train, new_dev = deterministic_split(expanded, dev_ratio=dev_ratio)
    if [s.id for s in new_dev] != [s.id for s in base_dev]:
        raise ValueError("Expanded manifest changed the ordered development split")
    if len(new_train) <= len(base_train):
        raise ValueError("Expanded manifest must contain additional training samples")

    base_vocabulary = CharacterVocabulary.from_texts([s.text for s in baseline])
    new_vocabulary = CharacterVocabulary.from_texts([s.text for s in expanded])
    if base_vocabulary.checksum() != new_vocabulary.checksum():
        raise ValueError("Matched training expansion must not change the vocabulary")

    train_texts = {expanded_ids[s.id][0] for s in new_train}
    dev_texts = {expanded_ids[s.id][0] for s in new_dev}
    if train_texts & dev_texts:
        raise ValueError("Training expansion has normalized label overlap with development")
    train_images = {expanded_ids[s.id][1] for s in new_train}
    dev_images = {expanded_ids[s.id][1] for s in new_dev}
    if train_images & dev_images:
        raise ValueError("Training expansion has image overlap with development")
    base_train_texts = {base_ids[s.id][0] for s in base_train}

    return {
        "schema_version": "1",
        "type": "matched-recognizer-training-expansion",
        "identity_strategy": "ordered-id-normalized-text-image-sha256-v1",
        "split_strategy": SPLIT_STRATEGY,
        "dev_ratio": dev_ratio,
        "baseline_train_samples": len(base_train),
        "expanded_train_samples": len(new_train),
        "fixed_dev_samples": len(base_dev),
        "added_train_samples": len(new_train) - len(base_train),
        "baseline_training_text_groups": len(base_train_texts),
        "expanded_training_text_groups": len(train_texts),
        "added_training_text_groups": len(train_texts - base_train_texts),
        "baseline_training_identity_sha256": _fingerprint(base_train, base_ids),
        "expanded_training_identity_sha256": _fingerprint(new_train, expanded_ids),
        "dev_identity_sha256": _fingerprint(base_dev, base_ids),
        "vocabulary_checksum": base_vocabulary.checksum(),
        "image_hashes_verified": True,
        "normalized_label_overlap": 0,
        "cross_split_image_overlap": 0,
        "document_source_separation": "unverified",
        "training_rights": "not-assessed",
    }


def audit_recognizer_holdout(
    excluded_manifests: list[list[TrainingSample]],
    evaluation: list[TrainingSample],
    vocabulary: CharacterVocabulary,
) -> dict[str, Any]:
    """Verify exact-label/image isolation against every supplied prior manifest.

    This is an explicit input audit, not proof of unseen documents, complete
    model-training provenance, source rights, or independent test performance.
    Repeated labels with distinct images are allowed and counted as one group.
    """
    if not excluded_manifests:
        raise ValueError("Holdout audit requires at least one exclusion manifest")
    if not evaluation or any(not samples for samples in excluded_manifests):
        raise ValueError("Holdout and exclusion manifests must be non-empty")

    excluded_texts: set[str] = set()
    excluded_images: set[str] = set()
    excluded_identities: list[dict[str, Any]] = []
    for samples in excluded_manifests:
        identities = _verified_identities(samples)
        excluded_texts.update(text for text, _ in identities.values())
        excluded_images.update(image for _, image in identities.values())
        excluded_identities.append({
            "samples": len(samples),
            "identity_sha256": _fingerprint(samples, identities),
        })

    identities = _verified_identities(evaluation)
    texts = {text for text, _ in identities.values()}
    images = {image for _, image in identities.values()}
    if texts & excluded_texts:
        raise ValueError("Evaluation has normalized label overlap with supplied exclusions")
    if images & excluded_images:
        raise ValueError("Evaluation has image overlap with supplied exclusions")
    if len(images) != len(evaluation):
        raise ValueError("Holdout contains a duplicate evaluation image")
    allowed = set(vocabulary.characters)
    if any(set(text) - allowed for text in texts):
        raise ValueError("Evaluation text contains characters outside the supplied vocabulary")

    return {
        "schema_version": "1",
        "type": "recognizer-holdout-audit",
        "identity_strategy": "ordered-id-normalized-text-image-sha256-v1",
        "evaluation_samples": len(evaluation),
        "evaluation_text_groups": len(texts),
        "evaluation_identity_sha256": _fingerprint(evaluation, identities),
        "excluded_manifest_count": len(excluded_manifests),
        "excluded_text_groups": len(excluded_texts),
        "excluded_unique_images": len(excluded_images),
        "excluded_manifests": excluded_identities,
        "vocabulary_checksum": vocabulary.checksum(),
        "vocabulary_size": vocabulary.size,
        "image_hashes_verified": True,
        "normalized_label_overlap": 0,
        "image_overlap": 0,
        "exclusion_coverage": "caller-supplied-manifests-only",
        "document_source_separation": "unverified",
        "source_rights": "not-assessed",
    }
