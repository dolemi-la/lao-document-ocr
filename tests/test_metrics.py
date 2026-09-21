from lao_document_ocr.metrics import character_error_rate, word_error_rate


def test_character_error_rate() -> None:
    assert character_error_rate("abc", "abc") == 0
    assert character_error_rate("abc", "axc") == 1 / 3


def test_word_error_rate() -> None:
    assert word_error_rate("one two", "one two") == 0
    assert word_error_rate("one two", "one three") == 0.5


def test_empty_reference() -> None:
    assert character_error_rate("", "") == 0
    assert character_error_rate("", "x") == 1
