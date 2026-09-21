import io

import pymupdf
from PIL import Image

from lao_document_ocr.embedded_images import extract_pdf_embedded_images


def _png_bytes() -> bytes:
    image = Image.new("RGB", (120, 80), (120, 150, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_extracts_smaller_native_pdf_image(tmp_path) -> None:
    path = tmp_path / "image.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page(width=600, height=800)
    page.insert_image(
        pymupdf.Rect(100, 200, 300, 360),
        stream=_png_bytes(),
        keep_proportion=False,
    )
    pdf.save(path)
    pdf.close()

    pdf = pymupdf.open(path)
    page = pdf[0]
    assets = extract_pdf_embedded_images(
        pdf,
        page,
        rendered_width=1200,
        rendered_height=1600,
    )
    pdf.close()

    assert len(assets) == 1
    asset = assets[0]
    assert asset.bbox.x == 200
    assert asset.bbox.y == 400
    assert asset.bbox.width == 400
    assert asset.bbox.height == 320
    assert asset.width_ratio == 1 / 3
    assert asset.png_bytes.startswith(b"\x89PNG")


def test_excludes_near_full_page_scan_image(tmp_path) -> None:
    path = tmp_path / "scan.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page(width=600, height=800)
    page.insert_image(
        pymupdf.Rect(0, 0, 600, 800),
        stream=_png_bytes(),
        keep_proportion=False,
    )
    pdf.save(path)
    pdf.close()

    pdf = pymupdf.open(path)
    assets = extract_pdf_embedded_images(
        pdf,
        pdf[0],
        rendered_width=1200,
        rendered_height=1600,
    )
    pdf.close()

    assert assets == []
