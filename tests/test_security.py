import pymupdf
import pytest
from PIL import Image

from services.api.app.security import (
    UploadValidationError,
    sanitize_filename,
    validate_uploaded_content,
)


def test_sanitize_filename_strips_paths_controls_and_unsafe_punctuation() -> None:
    assert sanitize_filename("../../hello.png") == "hello.png"
    assert sanitize_filename(r"..\hello.png") == "hello.png"
    assert sanitize_filename('my "scan"?.png') == "my _scan__.png"
    assert sanitize_filename("ສະບາຍດີ.png") == "ສະບາຍດີ.png"
    assert sanitize_filename("\x00\x01.png") == "document.png"


def test_validate_image_accepts_matching_content(tmp_path) -> None:
    path = tmp_path / "sample.png"
    Image.new("RGB", (40, 30), "white").save(path)

    validate_uploaded_content(
        path,
        ".png",
        max_pages=60,
        max_page_pixels=10_000,
    )


def test_validate_image_rejects_extension_mismatch(tmp_path) -> None:
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (40, 30), "white").save(path, format="PNG")

    with pytest.raises(UploadValidationError, match="does not match"):
        validate_uploaded_content(
            path,
            ".jpg",
            max_pages=60,
            max_page_pixels=10_000,
        )


def test_validate_image_rejects_pixel_limit(tmp_path) -> None:
    path = tmp_path / "sample.png"
    Image.new("RGB", (100, 100), "white").save(path)

    with pytest.raises(UploadValidationError, match="pixel limit"):
        validate_uploaded_content(
            path,
            ".png",
            max_pages=60,
            max_page_pixels=5_000,
        )


def test_validate_pdf_rejects_fake_pdf(tmp_path) -> None:
    path = tmp_path / "fake.pdf"
    path.write_bytes(b"not a pdf")

    with pytest.raises(UploadValidationError, match="does not match"):
        validate_uploaded_content(
            path,
            ".pdf",
            max_pages=60,
            max_page_pixels=1_000_000,
        )


def test_validate_pdf_rejects_page_limit(tmp_path) -> None:
    path = tmp_path / "pages.pdf"
    pdf = pymupdf.open()
    pdf.new_page(width=100, height=100)
    pdf.new_page(width=100, height=100)
    pdf.save(path)
    pdf.close()

    with pytest.raises(UploadValidationError, match="maximum is 1"):
        validate_uploaded_content(
            path,
            ".pdf",
            max_pages=1,
            max_page_pixels=1_000_000,
        )


def test_validate_pdf_rejects_encrypted_file(tmp_path) -> None:
    path = tmp_path / "encrypted.pdf"
    pdf = pymupdf.open()
    pdf.new_page(width=100, height=100)
    pdf.save(
        path,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
    )
    pdf.close()

    with pytest.raises(UploadValidationError, match="Encrypted PDFs"):
        validate_uploaded_content(
            path,
            ".pdf",
            max_pages=60,
            max_page_pixels=1_000_000,
        )
