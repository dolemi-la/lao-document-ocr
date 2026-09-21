
import pytest
from PIL import Image, ImageDraw

from lao_document_ocr.visual_fidelity import (
    compare_document_images,
    compare_page_images,
    find_office_binary,
    load_reference_pages,
    render_docx_pages,
)


def _page() -> Image.Image:
    image = Image.new("RGB", (600, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 80, 540, 130), fill="black")
    draw.rectangle((80, 220, 320, 420), fill=(80, 130, 210))
    return image


def test_identical_pages_score_one() -> None:
    page = _page()
    metrics = compare_page_images(page, page.copy())

    assert metrics.pixel_similarity == pytest.approx(1.0)
    assert metrics.foreground_iou == pytest.approx(1.0)
    assert metrics.edge_f1 == pytest.approx(1.0)
    assert metrics.composite_score == pytest.approx(1.0)


def test_shifted_content_scores_lower() -> None:
    reference = _page()
    prediction = Image.new("RGB", (600, 800), "white")
    prediction.paste(reference.crop((0, 0, 550, 760)), (40, 30))

    metrics = compare_page_images(reference, prediction)

    assert metrics.composite_score < 1.0
    assert metrics.foreground_iou < 1.0


def test_page_count_penalizes_document_score() -> None:
    page = _page()
    metrics = compare_document_images(
        [page, page],
        [page],
    )

    assert metrics.page_count_score == 0.5
    assert metrics.composite_score == pytest.approx(0.5)


def test_load_reference_image(tmp_path) -> None:
    path = tmp_path / "reference.png"
    _page().save(path)

    pages = load_reference_pages(path)

    assert len(pages) == 1
    assert pages[0].size == (600, 800)


def test_explicit_missing_office_binary_returns_none(tmp_path) -> None:
    missing = tmp_path / "missing-office"
    assert find_office_binary(missing) is None


def test_render_docx_requires_office_binary_when_unavailable(tmp_path, monkeypatch) -> None:
    docx = tmp_path / "sample.docx"
    docx.write_bytes(b"not important for missing-binary check")
    monkeypatch.setenv("PATH", "")

    with pytest.raises(RuntimeError, match="LibreOffice"):
        render_docx_pages(docx)
