from lao_document_ocr.normalization import normalize_lao_text


def test_normalization_uses_nfc_and_cleans_horizontal_space() -> None:
    source = "  Hello\t  world  \n\n\nNext "
    assert normalize_lao_text(source) == "Hello world\n\nNext"
