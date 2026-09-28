"""Fresh-evaluation audits verify supplied exclusions, not global independence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from lao_document_ocr.training_expansion import audit_recognizer_holdout
from lao_document_ocr.training_manifest import TrainingSample
from lao_document_ocr.vocabulary import CharacterVocabulary


def sample(root: Path, name: str, text: str, shade: int) -> TrainingSample:
    path = root / f"{name}.png"
    Image.new("L", (20, 16), shade).save(path)
    return TrainingSample(name, path, text, hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def inputs(tmp_path):
    excluded = [
        [sample(tmp_path, "private-train-id", "ab", 30)],
        [sample(tmp_path, "private-dev-id", "ba", 60)],
    ]
    holdout = [sample(tmp_path, "unseen-one", "aa", 90), sample(tmp_path, "unseen-two", "bb", 120)]
    return excluded, holdout, CharacterVocabulary.from_texts(["ab"])


def test_holdout_report_contains_only_counts_and_identities(inputs, tmp_path):
    excluded, holdout, vocabulary = inputs
    report = audit_recognizer_holdout(excluded, holdout, vocabulary)
    assert report["evaluation_samples"] == report["evaluation_text_groups"] == 2
    assert report["excluded_text_groups"] == 2
    assert report["excluded_manifest_count"] == 2
    assert report["normalized_label_overlap"] == report["image_overlap"] == 0
    assert report["image_hashes_verified"] is True
    assert report["vocabulary_checksum"] == vocabulary.checksum()
    assert report["document_source_separation"] == "unverified"
    assert report["exclusion_coverage"] == "caller-supplied-manifests-only"
    assert report["source_rights"] == "not-assessed"
    payload = json.dumps(report, allow_nan=False)
    for private in (
        "private-train-id",
        "private-dev-id",
        "unseen-one",
        "unseen-two",
        str(tmp_path),
        '"aa"',
        '"bb"',
        '"ab"',
        '"ba"',
    ):
        assert private not in payload
    assert len(report["evaluation_identity_sha256"]) == 64
    assert report == audit_recognizer_holdout(excluded, holdout, vocabulary)


def test_audit_allows_distinct_variants_counts_unique_labels(inputs, tmp_path):
    excluded, holdout, vocabulary = inputs
    holdout.append(sample(tmp_path, "third", "aa", 150))
    report = audit_recognizer_holdout(excluded, holdout, vocabulary)
    assert report["evaluation_samples"] == 3
    assert report["evaluation_text_groups"] == 2


def test_audit_checks_every_exclusion_and_normalizes(inputs):
    excluded, holdout, vocabulary = inputs
    excluded[-1] = [replace(excluded[-1][0], text="   aa \t")]
    with pytest.raises(ValueError, match="normalized label overlap"):
        audit_recognizer_holdout(excluded, holdout, vocabulary)


def test_audit_rejects_byte_identical_image_with_different_label(inputs):
    excluded, holdout, vocabulary = inputs
    excluded[-1] = [replace(holdout[0], id="other-id", text="ba")]
    with pytest.raises(ValueError, match="image overlap"):
        audit_recognizer_holdout(excluded, holdout, vocabulary)


def test_audit_rejects_duplicate_evaluation_images(inputs):
    excluded, holdout, vocabulary = inputs
    holdout.append(replace(holdout[0], id="duplicate-image", text="bb"))
    with pytest.raises(ValueError, match="duplicate evaluation image"):
        audit_recognizer_holdout(excluded, holdout, vocabulary)


@pytest.mark.parametrize("which", ["evaluation", "exclusion"])
def test_audit_rechecks_bytes_for_both_sides(inputs, which):
    excluded, holdout, vocabulary = inputs
    chosen = holdout[0] if which == "evaluation" else excluded[-1][0]
    chosen.image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit_recognizer_holdout(excluded, holdout, vocabulary)


@pytest.mark.parametrize("bad_hash", [None, "", "0" * 63, "Z" * 64])
def test_audit_requires_pinned_image_hashes(inputs, bad_hash):
    excluded, holdout, vocabulary = inputs
    holdout[0] = replace(holdout[0], sha256=bad_hash)
    with pytest.raises(ValueError, match="pinned SHA-256"):
        audit_recognizer_holdout(excluded, holdout, vocabulary)


@pytest.mark.parametrize("which", ["evaluation", "exclusion"])
def test_audit_rejects_empty_manifests(inputs, which):
    excluded, holdout, vocabulary = inputs
    if which == "evaluation":
        holdout = []
    else:
        excluded[-1] = []
    with pytest.raises(ValueError, match="non-empty"):
        audit_recognizer_holdout(excluded, holdout, vocabulary)


def test_audit_requires_at_least_one_exclusion(inputs):
    _, holdout, vocabulary = inputs
    with pytest.raises(ValueError, match="at least one exclusion"):
        audit_recognizer_holdout([], holdout, vocabulary)


def test_audit_rejects_unknown_vocabulary_instead_of_filtering(inputs):
    excluded, holdout, vocabulary = inputs
    holdout[0] = replace(holdout[0], text="c")
    with pytest.raises(ValueError, match="outside the supplied vocabulary"):
        audit_recognizer_holdout(excluded, holdout, vocabulary)


def test_evaluation_order_is_part_of_identity(inputs):
    excluded, holdout, vocabulary = inputs
    first = audit_recognizer_holdout(excluded, holdout, vocabulary)
    second = audit_recognizer_holdout(excluded, list(reversed(holdout)), vocabulary)
    assert first["evaluation_identity_sha256"] != second["evaluation_identity_sha256"]


def test_exclusion_namespaces_may_overlap_without_hiding_labels(inputs):
    excluded, holdout, vocabulary = inputs
    excluded[-1][0] = replace(excluded[-1][0], id=excluded[0][0].id)
    report = audit_recognizer_holdout(excluded, holdout, vocabulary)
    assert report["excluded_text_groups"] == 2


def test_duplicate_ids_within_evaluation_are_rejected(inputs):
    excluded, holdout, vocabulary = inputs
    holdout[1] = replace(holdout[1], id=holdout[0].id)
    with pytest.raises(ValueError, match="Duplicate training sample ID"):
        audit_recognizer_holdout(excluded, holdout, vocabulary)


def write_manifest(path, samples):
    path.write_text(
        "".join(
            json.dumps({"id": s.id, "image": str(s.image), "text": s.text, "sha256": s.sha256})
            + "\n"
            for s in samples
        )
    )


def test_cli_repeatable_exclusions_and_no_overwrite(inputs, tmp_path, monkeypatch, capsys):
    import sys

    from lao_document_ocr.cli import main

    excluded, holdout, vocabulary = inputs
    paths = [tmp_path / "train.jsonl", tmp_path / "dev.jsonl"]
    for path, items in zip(paths, excluded, strict=True):
        write_manifest(path, items)
    candidate = tmp_path / "evaluation.jsonl"
    write_manifest(candidate, holdout)
    vocab_path = vocabulary.save(tmp_path / "vocab.json")
    output = tmp_path / "audit" / "report.json"
    argv = [
        "lao-ocr",
        "audit-recognizer-holdout",
        "--manifest",
        str(candidate),
        "--vocabulary",
        str(vocab_path),
        "--output",
        str(output),
    ]
    for path in paths:
        argv += ["--exclude-manifest", str(path)]
    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 0
    original = output.read_bytes()
    assert json.loads(original)["excluded_manifest_count"] == 2
    assert main() == 1
    assert "already exists" in capsys.readouterr().err
    assert output.read_bytes() == original


def test_cli_validation_failure_creates_no_output(inputs, tmp_path, monkeypatch):
    import sys

    from lao_document_ocr.cli import main

    excluded, holdout, vocabulary = inputs
    excluded[0][0] = replace(excluded[0][0], text="aa")
    excluded_path = tmp_path / "used.jsonl"
    write_manifest(excluded_path, excluded[0])
    candidate = tmp_path / "eval.jsonl"
    write_manifest(candidate, holdout)
    output = tmp_path / "must-not-exist" / "report.json"
    vocabulary.save(tmp_path / "vocab.json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lao-ocr",
            "audit-recognizer-holdout",
            "--manifest",
            str(candidate),
            "--exclude-manifest",
            str(excluded_path),
            "--vocabulary",
            str(tmp_path / "vocab.json"),
            "--output",
            str(output),
        ],
    )
    assert main() == 1
    assert not output.parent.exists()


def test_holdout_rejects_unicode_equivalent_excluded_label(inputs):
    excluded, holdout, _ = inputs
    excluded[0][0] = replace(excluded[0][0], text="é")
    holdout[0] = replace(holdout[0], text="e" + chr(0x301))
    vocabulary = CharacterVocabulary.from_texts(["abé"])
    with pytest.raises(ValueError, match="normalized label overlap"):
        audit_recognizer_holdout(excluded, holdout, vocabulary)


def test_holdout_rejects_empty_normalized_text(inputs):
    excluded, holdout, vocabulary = inputs
    holdout[0] = replace(holdout[0], text="   ")
    with pytest.raises(ValueError, match="empty normalized text"):
        audit_recognizer_holdout(excluded, holdout, vocabulary)


def test_exclusion_manifest_rejects_duplicate_ids(inputs):
    excluded, holdout, vocabulary = inputs
    excluded[0].append(replace(excluded[0][0], text="bb"))
    with pytest.raises(ValueError, match="Duplicate training sample ID"):
        audit_recognizer_holdout(excluded, holdout, vocabulary)
