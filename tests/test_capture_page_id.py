from PIL import Image

from lao_document_ocr.capture_page_id import (
    decode_page_id,
    decode_page_id_image,
    page_id_payload,
    render_page_id_qr,
)


def test_page_id_qr_round_trip() -> None:
    page_id = "project-authored-lao-v1-two-column-p0001"
    qr = render_page_id_qr(page_id, size_px=128)

    assert decode_page_id_image(qr) == page_id
    assert page_id_payload(page_id).endswith(page_id)


def test_page_id_qr_survives_page_scale_and_margin(tmp_path) -> None:
    page_id = "project-authored-lao-v1-receipt-p0007"
    qr = render_page_id_qr(page_id, size_px=140)
    page = Image.new("RGB", (1200, 1700), "white")
    page.paste(qr, (930, 90))

    capture = Image.new("RGB", (1500, 2100), (220, 215, 205))
    resized = page.resize((1080, 1530), Image.Resampling.BICUBIC)
    capture.paste(resized, (210, 260))
    path = tmp_path / "phone.jpg"
    capture.save(path, quality=82)

    assert decode_page_id(path) == page_id


def test_non_qr_image_returns_none() -> None:
    assert decode_page_id_image(
        Image.new("RGB", (500, 400), "white")
    ) is None


def test_invalid_empty_page_id_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="page_id"):
        render_page_id_qr("", size_px=100)
