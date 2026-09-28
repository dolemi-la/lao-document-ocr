"""Matched training expansions must not silently change the evaluation set."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from lao_document_ocr.training_expansion import audit_training_expansion
from lao_document_ocr.training_manifest import TrainingSample, deterministic_split


def _sample(tmp_path, index: int, text: str | None = None) -> TrainingSample:
    path = tmp_path / f"sample-{index}.png"
    Image.new("L", (40, 16), index % 256).save(path)
    return TrainingSample(
        id=f"sample-{index}",
        image=path,
        text=text or f"sample {index % 10}",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _fixture(tmp_path):
    base = [_sample(tmp_path, i) for i in range(20)]
    train, dev = deterministic_split(base, dev_ratio=0.3)
    addition = _sample(tmp_path, 100, train[0].text)
    return base, base + [addition], train, dev


def test_matched_expansion_is_fingerprinted_without_text_or_paths(tmp_path):
    base, expanded, train, dev = _fixture(tmp_path)
    report = audit_training_expansion(base, expanded, dev_ratio=0.3)
    assert report["baseline_train_samples"] == len(train)
    assert report["expanded_train_samples"] == len(train) + 1
    assert report["fixed_dev_samples"] == len(dev)
    assert report["added_train_samples"] == 1
    assert report["added_training_text_groups"] == 0
    assert report["image_hashes_verified"] is True
    assert report["normalized_label_overlap"] == report["cross_split_image_overlap"] == 0
    for field in (
        "dev_identity_sha256",
        "baseline_training_identity_sha256",
        "expanded_training_identity_sha256",
        "vocabulary_checksum",
    ):
        assert len(report[field]) == 64
    payload = json.dumps(report, allow_nan=False)
    assert str(tmp_path) not in payload
    assert not any(sample.id in payload or sample.text in payload for sample in base)
    assert report == audit_training_expansion(base, expanded, dev_ratio=0.3)


@pytest.mark.parametrize("side", ["baseline", "expanded"])
def test_duplicate_ids_are_rejected(tmp_path, side):
    base, expanded, _, _ = _fixture(tmp_path)
    if side == "baseline":
        base.append(base[0])
    else:
        expanded.append(expanded[0])
    with pytest.raises(ValueError, match="Duplicate"):
        audit_training_expansion(base, expanded, dev_ratio=0.3)


def test_missing_original_sample_is_rejected(tmp_path):
    base, expanded, train, _ = _fixture(tmp_path)
    expanded = [s for s in expanded if s.id != train[0].id]
    with pytest.raises(ValueError, match="original"):
        audit_training_expansion(base, expanded, dev_ratio=0.3)


@pytest.mark.parametrize("change", ["image", "label"])
def test_original_identity_changes_are_rejected(tmp_path, change):
    base, expanded, train, _ = _fixture(tmp_path)
    original = train[0]
    replacement = (
        replace(original, image=expanded[-1].image, sha256=expanded[-1].sha256)
        if change == "image"
        else replace(original, text=train[-1].text + " 1")
    )
    expanded = [replacement if s.id == original.id else s for s in expanded]
    with pytest.raises(ValueError, match="original"):
        audit_training_expansion(base, expanded, dev_ratio=0.3)


def test_new_dev_variants_are_not_a_matched_expansion(tmp_path):
    base, _, _, dev = _fixture(tmp_path)
    expanded = base + [_sample(tmp_path, 101, dev[0].text)]
    with pytest.raises(ValueError, match="development"):
        audit_training_expansion(base, expanded, dev_ratio=0.3)


def test_reordered_dev_is_rejected(tmp_path):
    base, expanded, _, dev = _fixture(tmp_path)
    assert len(dev) >= 2
    first, second = (expanded.index(dev[i]) for i in (0, 1))
    expanded[first], expanded[second] = expanded[second], expanded[first]
    with pytest.raises(ValueError, match="development"):
        audit_training_expansion(base, expanded, dev_ratio=0.3)


def test_label_group_fallback_cannot_move_baseline_dev(tmp_path):
    # Find a two-group all-training-bucket split. Its fallback moves one group to dev.
    # Adding a true dev group must not silently put that original holdout in train.
    pool = [_sample(tmp_path, i, f"candidate {i}") for i in range(40)]
    train, dev = deterministic_split(pool, dev_ratio=0.1)
    base = train[:2]
    with pytest.raises(ValueError, match="development"):
        audit_training_expansion(base, base + dev[:1], dev_ratio=0.1)


def test_cross_split_duplicate_image_is_rejected_even_with_different_label(tmp_path):
    base, expanded, train, dev = _fixture(tmp_path)
    expanded[-1] = replace(
        expanded[-1], image=dev[0].image, sha256=dev[0].sha256, text=train[0].text
    )
    with pytest.raises(ValueError, match="image overlap"):
        audit_training_expansion(base, expanded, dev_ratio=0.3)


def test_vocabulary_change_is_rejected(tmp_path):
    base, _, _, _ = _fixture(tmp_path)
    for i in range(100):
        new = _sample(tmp_path, 100, f"Z{i}")
        candidate = base + [new]
        _, before = deterministic_split(base, dev_ratio=0.3)
        _, after = deterministic_split(candidate, dev_ratio=0.3)
        if [s.id for s in before] == [s.id for s in after]:
            break
    with pytest.raises(ValueError, match="vocabulary"):
        audit_training_expansion(base, candidate, dev_ratio=0.3)


def test_image_hashes_are_verified_again_not_trusted(tmp_path):
    base, expanded, _, _ = _fixture(tmp_path)
    expanded[-1].image.write_bytes(b"changed image bytes")
    with pytest.raises(ValueError, match="SHA-256"):
        audit_training_expansion(base, expanded, dev_ratio=0.3)


def test_no_growth_is_rejected(tmp_path):
    base, _, _, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="additional training"):
        audit_training_expansion(base, base, dev_ratio=0.3)


def test_unpinned_image_is_rejected(tmp_path):
    base, expanded, _, _ = _fixture(tmp_path)
    expanded[-1] = replace(expanded[-1], sha256=None)
    with pytest.raises(ValueError, match="pinned SHA-256"):
        audit_training_expansion(base, expanded, dev_ratio=0.3)


def _write_manifest(path: Path, samples):
    path.write_text(
        "".join(
            json.dumps(
                {
                    "id": s.id,
                    "image": str(s.image),
                    "text": s.text,
                    "sha256": s.sha256,
                }
            )
            + "\n"
            for s in samples
        ),
        encoding="utf-8",
    )


def test_cli_audit_and_no_overwrite(tmp_path, monkeypatch, capsys):
    from lao_document_ocr.cli import main

    base, expanded, _, _ = _fixture(tmp_path)
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "expanded.jsonl"
    output = tmp_path / "audit.json"
    _write_manifest(baseline, base)
    _write_manifest(candidate, expanded)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "audit-training-expansion",
            "--baseline-manifest",
            str(baseline),
            "--expanded-manifest",
            str(candidate),
            "--dev-ratio",
            "0.3",
            "--output",
            str(output),
        ],
    )
    assert main() == 0
    assert json.loads(output.read_text())["added_train_samples"] == 1
    original = output.read_bytes()
    assert main() == 1
    assert output.read_bytes() == original
    assert "already exists" in capsys.readouterr().err


def test_cli_failure_does_not_create_report(tmp_path, monkeypatch):
    from lao_document_ocr.cli import main

    base, _, _, _ = _fixture(tmp_path)
    manifest = tmp_path / "baseline.jsonl"
    _write_manifest(manifest, base)
    output = tmp_path / "uncreated" / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "audit-training-expansion",
            "--baseline-manifest",
            str(manifest),
            "--expanded-manifest",
            str(manifest),
            "--dev-ratio",
            "0.3",
            "--output",
            str(output),
        ],
    )
    assert main() == 1
    assert not output.parent.exists()


def test_normalization_equivalent_labels_do_not_change_identity(tmp_path):
    base, expanded, _, _ = _fixture(tmp_path)
    expected = audit_training_expansion(base, expanded, dev_ratio=0.3)
    changed = [replace(s, text="  " + s.text.replace(" ", "   ") + "  ") for s in expanded]
    actual = audit_training_expansion(base, changed, dev_ratio=0.3)
    assert actual == expected
    assert all(not s.text.startswith("  ") for s in expanded)


def test_new_training_text_group_is_counted(tmp_path):
    base, _, _, dev = _fixture(tmp_path)
    for index in range(100, 200):
        added = _sample(tmp_path, index, f"sample {index}")
        candidate = base + [added]
        _, candidate_dev = deterministic_split(candidate, dev_ratio=0.3)
        if [s.id for s in candidate_dev] == [s.id for s in dev]:
            break
    report = audit_training_expansion(base, candidate, dev_ratio=0.3)
    assert report["added_training_text_groups"] == 1
    assert report["added_train_samples"] == 1


@pytest.mark.parametrize("checksum", ["", "bad", "0" * 63, "A" * 64, True])
def test_malformed_hash_pin_is_rejected(tmp_path, checksum):
    base, expanded, _, _ = _fixture(tmp_path)
    expanded[-1] = replace(expanded[-1], sha256=checksum)
    with pytest.raises(ValueError, match="pinned SHA-256"):
        audit_training_expansion(base, expanded, dev_ratio=0.3)


@pytest.mark.parametrize("ratio", [0, 1, -0.5, float("nan"), float("inf")])
def test_expansion_rejects_invalid_ratio(tmp_path, ratio):
    base, expanded, _, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="dev_ratio"):
        audit_training_expansion(base, expanded, dev_ratio=ratio)
