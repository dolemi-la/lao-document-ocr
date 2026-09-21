import pytest

from lao_document_ocr.vocabulary import BLANK_ID, CharacterVocabulary


def test_build_encode_and_ctc_decode() -> None:
    vocab = CharacterVocabulary.from_texts(["ສະບາຍດີ", "OCR 123"])
    encoded = vocab.encode("OCR")
    assert all(token > BLANK_ID for token in encoded)

    # CTC: repeated token collapses unless separated by blank.
    first = encoded[0]
    second = encoded[1]
    assert vocab.decode_ctc([first, first, 0, first, second]) == "OOC"


def test_vocabulary_round_trip(tmp_path) -> None:
    vocab = CharacterVocabulary.from_texts(["ລາວ OCR"])
    path = vocab.save(tmp_path / "vocab.json")
    loaded = CharacterVocabulary.load(path)
    assert loaded == vocab
    assert loaded.checksum() == vocab.checksum()
    assert loaded.size == len(loaded.characters) + 1


def test_encode_rejects_unknown_character() -> None:
    vocab = CharacterVocabulary.from_texts(["abc"])
    with pytest.raises(ValueError, match="missing from vocabulary"):
        vocab.encode("abcລ")
