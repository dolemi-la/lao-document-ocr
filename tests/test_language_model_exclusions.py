from __future__ import annotations

import json
import sys

import pytest

from lao_document_ocr.cli import main
from lao_document_ocr.language_model import (
    CharacterNgramLanguageModel,
    save_language_model,
    train_character_ngram_language_model,
)
from lao_document_ocr.vocabulary import CharacterVocabulary


def test_exclusions_remove_every_normalized_duplicate_before_learning() -> None:
    held_out = "ສະບາຍດີ ລາວ"
    lines = ["train", "  ສະບາຍດີ   ລາວ  ", held_out, "e\u0301", "é"]
    vocabulary = CharacterVocabulary.from_texts(lines)
    model, stats = train_character_ngram_language_model(
        lines, vocabulary, exclude_texts=[held_out, "é"],
    )
    baseline, _ = train_character_ngram_language_model(["train"], vocabulary)

    assert model.counts == baseline.counts
    assert model.totals == baseline.totals
    assert stats.total_lines == 5
    assert stats.used_lines == 1
    assert stats.excluded_lines == 4
    assert stats.skipped_lines == 0
    assert model.text_exclusions["unique_texts"] == 2
    assert model.text_exclusions["excluded_lines"] == 4
    assert model.text_exclusions["strategy"] == "normalized-exact-text-v1"
    assert len(model.text_exclusions["sha256"]) == 64
    assert held_out not in json.dumps(model.text_exclusions, ensure_ascii=False)


def test_exclusion_fingerprint_is_order_and_duplicate_independent() -> None:
    vocabulary = CharacterVocabulary.from_texts(["train", "a", "b"])
    first, _ = train_character_ngram_language_model(
        ["train", "a", "b"], vocabulary, exclude_texts=[" a ", "b", "a"],
    )
    second, _ = train_character_ngram_language_model(
        ["train", "a", "b"], vocabulary, exclude_texts=["b", "a"],
    )
    different, _ = train_character_ngram_language_model(
        ["train", "a", "b"], vocabulary, exclude_texts=["a"],
    )
    assert first.text_exclusions == second.text_exclusions
    assert first.text_exclusions["sha256"] != different.text_exclusions["sha256"]


def test_nonmatching_exclusions_do_not_claim_that_no_guard_was_used() -> None:
    vocabulary = CharacterVocabulary.from_texts(["train"])
    model, stats = train_character_ngram_language_model(
        ["train"], vocabulary, exclude_texts=["unseen text"],
    )
    assert stats.used_lines == 1
    assert stats.excluded_lines == 0
    assert model.text_exclusions["unique_texts"] == 1


@pytest.mark.parametrize("excluded", [[], ["", "  ", "\t\n"]])
def test_explicit_empty_exclusions_are_rejected(excluded) -> None:
    vocabulary = CharacterVocabulary.from_texts(["train"])
    with pytest.raises(ValueError, match="exclusion.*non-empty"):
        train_character_ngram_language_model(
            ["train"], vocabulary, exclude_texts=excluded,
        )


def test_excluding_all_encodable_lines_fails() -> None:
    vocabulary = CharacterVocabulary.from_texts(["held out"])
    with pytest.raises(ValueError, match="No corpus lines"):
        train_character_ngram_language_model(
            ["held out", " held  out "], vocabulary, exclude_texts=["held out"],
        )


def test_exclusions_round_trip_and_remain_in_runtime_metadata(tmp_path) -> None:
    vocabulary = CharacterVocabulary.from_texts(["train", "dev"])
    model, _ = train_character_ngram_language_model(
        ["train", "dev"], vocabulary, exclude_texts=["dev"],
    )
    path = save_language_model(model, tmp_path / "model.json")
    loaded = CharacterNgramLanguageModel.load(path)
    assert loaded.text_exclusions == model.text_exclusions
    assert loaded.metadata()["text_exclusions"] == model.text_exclusions
    assert loaded.counts == model.counts
    assert loaded.training_stats["excluded_lines"] == 1


def test_legacy_language_model_does_not_claim_exclusion_provenance(tmp_path) -> None:
    vocabulary = CharacterVocabulary.from_texts(["train"])
    model, _ = train_character_ngram_language_model(["train"], vocabulary)
    path = save_language_model(model, tmp_path / "legacy.json")
    payload = json.loads(path.read_text())
    payload.pop("text_exclusions", None)
    payload["training_stats"].pop("excluded_lines", None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = CharacterNgramLanguageModel.load(path)
    assert loaded.text_exclusions is None
    assert loaded.metadata().get("text_exclusions") is None


@pytest.mark.parametrize("field,value", [
    ("strategy", "unverified"),
    ("sha256", "not-a-checksum"),
    ("unique_texts", 0),
    ("unique_texts", True),
    ("excluded_lines", -1),
])
def test_invalid_exclusion_metadata_is_rejected(tmp_path, field, value) -> None:
    vocabulary = CharacterVocabulary.from_texts(["train", "dev"])
    model, _ = train_character_ngram_language_model(
        ["train", "dev"], vocabulary, exclude_texts=["dev"],
    )
    path = save_language_model(model, tmp_path / "invalid.json")
    payload = json.loads(path.read_text())
    payload["text_exclusions"][field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exclusion"):
        CharacterNgramLanguageModel.load(path)


def _cli_args(tmp_path, corpus_text):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(corpus_text, encoding="utf-8")
    vocabulary_path = tmp_path / "vocab.json"
    CharacterVocabulary.from_texts([corpus_text]).save(vocabulary_path)
    output = tmp_path / "language-model.json"
    return [
        "lao-ocr", "train-char-lm", "--corpus", str(corpus),
        "--vocabulary", str(vocabulary_path), "--output", str(output),
    ], output


def test_cli_accepts_multiple_exclusion_corpora(tmp_path, monkeypatch, capsys) -> None:
    args, output = _cli_args(tmp_path, "train\ndev\n  dev  \ntest\n")
    dev = tmp_path / "dev.txt"
    test = tmp_path / "test.txt"
    dev.write_text("dev\n", encoding="utf-8")
    test.write_text("test\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", args + [
        "--exclude-corpus", str(dev), "--exclude-corpus", str(test),
    ])
    assert main() == 0
    model = CharacterNgramLanguageModel.load(output)
    assert model.training_stats["used_lines"] == 1
    assert model.training_stats["excluded_lines"] == 3
    assert model.text_exclusions["unique_texts"] == 2
    assert not capsys.readouterr().err


@pytest.mark.parametrize("failure", ["missing", "empty", "all"])
def test_cli_failed_exclusion_preserves_existing_output(
    tmp_path, monkeypatch, capsys, failure,
) -> None:
    args, output = _cli_args(tmp_path, "train\n")
    output.write_bytes(b"existing artifact")
    exclusion = tmp_path / "exclude.txt"
    if failure == "empty":
        exclusion.write_text("\n  \n", encoding="utf-8")
    elif failure == "all":
        exclusion.write_text("train\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", args + ["--exclude-corpus", str(exclusion)])
    assert main() == 1
    assert capsys.readouterr().err
    assert output.read_bytes() == b"existing artifact"


def test_cli_without_exclusions_warns_that_separation_is_unverified(
    tmp_path, monkeypatch, capsys,
) -> None:
    args, output = _cli_args(tmp_path, "train\n")
    monkeypatch.setattr(sys, "argv", args)
    assert main() == 0
    assert "unverified" in capsys.readouterr().err
    assert CharacterNgramLanguageModel.load(output).text_exclusions is None
