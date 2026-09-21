from PIL import Image, ImageDraw

from lao_document_ocr.line_detection import detect_text_lines


def test_detect_text_lines_finds_separate_rows() -> None:
    image = Image.new("L", (500, 220), 255)
    draw = ImageDraw.Draw(image)

    for x in (40, 110, 200, 300):
        draw.rectangle((x, 45, x + 45, 65), fill=0)
    for x in (55, 140, 250):
        draw.rectangle((x, 135, x + 60, 158), fill=0)

    boxes = detect_text_lines(image)

    assert len(boxes) == 2
    assert boxes[0].y < boxes[1].y
    assert boxes[0].width > 300
    assert boxes[1].width > 250


def test_detect_text_lines_blank_page() -> None:
    image = Image.new("L", (300, 200), 255)
    assert detect_text_lines(image) == []
