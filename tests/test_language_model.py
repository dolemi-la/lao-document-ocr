import math

import pytest

from lao_document_ocr.language_model import (
    CharacterNgramLanguageModel,
    save_language_model,
    train_character_ngram_language_model,
)
from lao_document_ocr.vocabulary import CharacterVocabulary


def test_trigram_prefers_observed_continuation() -> None:
    vocab = CharacterVocabulary.from_texts(["abac"])
    model, stats = train_character_ngram_language_model(
        ["abab", "abab", "abac"],
        vocab,
        order=3,
        alpha=0.1,
    )
    mapping = vocab.char_to_id
    prefix = (mapping["a"], mapping["b"], mapping["a"])

    score_b = model.score_extension(prefix, mapping["b"])
    score_c = model.score_extension(prefix, mapping["c"])

    assert score_b > score_c
    assert stats.used_lines == 3
    assert stats.skipped_lines == 0


def test_unseen_context_backs_off_to_shorter_context() -> None:
    vocab = CharacterVocabulary.from_texts(["abcx"])
    model, _ = train_character_ngram_language_model(
        ["ab", "ac", "ax"],
        vocab,
        order=3,
        alpha=0.1,
    )
    mapping = vocab.char_to_id

    score = model.score_extension((mapping["c"], mapping["c"]), mapping["a"])

    assert math.isfinite(score)


def test_training_skips_lines_with_unknown_characters() -> None:
    vocab = CharacterVocabulary.from_texts(["ab"])
    _, stats = train_character_ngram_language_model(
        ["ab", "aZ"],
        vocab,
        order=2,
    )

    assert stats.total_lines == 2
    assert stats.used_lines == 1
    assert stats.skipped_lines == 1


def test_language_model_round_trip_and_vocab_checksum(tmp_path) -> None:
    vocab = CharacterVocabulary.from_texts(["abc"])
    model, _ = train_character_ngram_language_model(
        ["abc", "aba"],
        vocab,
        order=3,
    )
    path = save_language_model(model, tmp_path / "lm.json")
    loaded = CharacterNgramLanguageModel.load(path)

    assert loaded.vocabulary_checksum == vocab.checksum()
    assert loaded.order == 3
    assert loaded.artifact_sha256 is not None
    assert loaded.score_sequence(tuple(vocab.encode("abc"))) == pytest.approx(
        model.score_sequence(tuple(vocab.encode("abc")))
    )


def test_training_rejects_empty_usable_corpus() -> None:
    vocab = CharacterVocabulary.from_texts(["ab"])
    with pytest.raises(ValueError, match="No corpus lines"):
        train_character_ngram_language_model(["ZZ"], vocab)
