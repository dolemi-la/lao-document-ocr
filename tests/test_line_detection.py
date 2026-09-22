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


def test_region_aware_lines_keep_two_columns_separate() -> None:
    from lao_document_ocr.line_detection import detect_region_aware_lines

    image = Image.new("L", (800, 600), 255)
    draw = ImageDraw.Draw(image)
    for y in (80, 125, 170):
        for x in (50, 105, 165):
            draw.rectangle((x, y, x + 34, y + 16), fill=0)
        for x in (470, 525, 585):
            draw.rectangle((x, y, x + 34, y + 16), fill=0)

    boxes = detect_region_aware_lines(image)

    assert len(boxes) == 6
    left = [box for box in boxes if box.x < 300]
    right = [box for box in boxes if box.x > 350]
    assert len(left) == 3
    assert len(right) == 3
    assert all(box.width < 250 for box in boxes)


def test_region_aware_lines_handle_heading_plus_columns() -> None:
    from lao_document_ocr.line_detection import detect_region_aware_lines

    image = Image.new("L", (800, 700), 255)
    draw = ImageDraw.Draw(image)
    for x in (70, 145, 225, 305, 385, 465):
        draw.rectangle((x, 40, x + 34, 56), fill=0)
    for y in (150, 195, 240):
        for x in (50, 105, 165):
            draw.rectangle((x, y, x + 34, y + 16), fill=0)
        for x in (470, 525, 585):
            draw.rectangle((x, y, x + 34, y + 16), fill=0)

    boxes = detect_region_aware_lines(image)

    assert len(boxes) == 7
    heading = min(boxes, key=lambda box: box.y)
    assert heading.width > 400
    assert heading.y < 80
