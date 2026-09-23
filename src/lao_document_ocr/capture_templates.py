from __future__ import annotations

import math
import unicodedata
from enum import StrEnum
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from lao_document_ocr.capture_page_id import render_page_id_qr
from lao_document_ocr.normalization import normalize_lao_text


class CaptureTemplate(StrEnum):
    PLAIN = "plain"
    TWO_COLUMN = "two-column"
    RULED_TABLE = "ruled-table"
    BORDERLESS_TABLE = "borderless-table"
    RECEIPT = "receipt"
    FORM = "form"


def _clusters(text: str) -> list[str]:
    clusters: list[str] = []
    for char in text:
        if not clusters or not (
            unicodedata.combining(char)
            or unicodedata.category(char).startswith("M")
        ):
            clusters.append(char)
        else:
            clusters[-1] += char
    return clusters


def _fits(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    width: int,
) -> bool:
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return (right - left) <= width


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    width: int,
) -> list[str]:
    text = normalize_lao_text(text)
    if not text:
        return []

    words = text.split(" ")
    lines: list[str] = []
    current = ""

    for word in words:
        candidate = word if not current else f"{current} {word}"
        if _fits(draw, candidate, font, width):
            current = candidate
            continue

        if current:
            lines.append(current)
            current = ""

        if _fits(draw, word, font, width):
            current = word
            continue

        clusters = _clusters(word)
        cluster_line: list[str] = []
        for cluster in clusters:
            candidate = "".join([*cluster_line, cluster])
            if cluster_line and not _fits(draw, candidate, font, width):
                lines.append("".join(cluster_line))
                cluster_line = [cluster]
            else:
                cluster_line.append(cluster)
        if cluster_line:
            current = "".join(cluster_line)

    if current:
        lines.append(current)
    return lines


def _fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: Path,
    *,
    start_size: int,
    min_size: int,
    width: int,
) -> ImageFont.FreeTypeFont:
    size = start_size
    while size > min_size:
        font = ImageFont.truetype(str(font_path), size=size)
        if _fits(draw, text, font, width):
            return font
        size -= 1
    return ImageFont.truetype(str(font_path), size=min_size)


def _language_tags(lines: list[str]) -> set[str]:
    joined = "\n".join(lines)
    has_lao = any("຀" <= char <= "໿" for char in joined)
    has_latin = any(
        ("A" <= char <= "Z") or ("a" <= char <= "z")
        for char in joined
    )
    if has_lao and has_latin:
        return {"language:mixed"}
    if has_lao:
        return {"language:lao"}
    if has_latin:
        return {"language:latin"}
    return {"language:other"}


def template_tags(
    template: CaptureTemplate,
    source_lines: list[str],
) -> tuple[str, ...]:
    tags = {
        "source:capture-pack",
        "page-id:qr-v1",
        f"template:{template.value}",
        *_language_tags(source_lines),
    }
    if template == CaptureTemplate.PLAIN:
        tags.add("layout:plain")
    elif template == CaptureTemplate.TWO_COLUMN:
        tags.add("layout:multi-column")
    elif template == CaptureTemplate.RULED_TABLE:
        tags.update({"layout:table", "content:table", "table:ruled"})
    elif template == CaptureTemplate.BORDERLESS_TABLE:
        tags.update({"layout:table", "content:table", "table:borderless"})
    elif template == CaptureTemplate.RECEIPT:
        tags.update({"layout:receipt", "document:receipt"})
    elif template == CaptureTemplate.FORM:
        tags.update({"layout:form", "document:form"})
    return tuple(sorted(tags))


def _base(
    page_id: str,
    font_path: Path,
    dpi: int,
) -> tuple[
    Image.Image,
    ImageDraw.ImageDraw,
    ImageFont.FreeTypeFont,
    ImageFont.FreeTypeFont,
    int,
    int,
    int,
    int,
]:
    width = round(8.27 * dpi)
    height = round(11.69 * dpi)
    margin = round(0.70 * dpi)
    body_font = ImageFont.truetype(
        str(font_path),
        size=max(18, round(dpi * 0.115)),
    )
    meta_font = ImageFont.truetype(
        str(font_path),
        size=max(13, round(dpi * 0.072)),
    )
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    y = margin
    draw.text(
        (margin, y),
        "Lao OCR Capture Pack",
        font=meta_font,
        fill="black",
    )
    y += round(dpi * 0.20)
    draw.text(
        (margin, y),
        f"Page ID: {page_id}",
        font=meta_font,
        fill="black",
    )

    qr = render_page_id_qr(
        page_id,
        size_px=max(64, round(dpi * 0.82)),
    )
    qr_x = width - margin - qr.width
    qr_y = margin
    image.paste(qr, (qr_x, qr_y))

    text_body_top = y + round(dpi * 0.35)
    qr_body_top = qr_y + qr.height + round(dpi * 0.10)
    y = max(text_body_top, qr_body_top)
    return image, draw, body_font, meta_font, width, height, margin, y


def _footer(
    draw: ImageDraw.ImageDraw,
    page_id: str,
    meta_font: ImageFont.FreeTypeFont,
    *,
    height: int,
    margin: int,
) -> str:
    footer = f"Capture ID: {page_id}"
    draw.text(
        (margin, height - margin),
        footer,
        font=meta_font,
        fill="black",
        anchor="ls",
    )
    return footer


def _truth(
    page_id: str,
    content_lines: list[str],
    footer: str,
) -> str:
    return normalize_lao_text(
        "\n".join(
            [
                "Lao OCR Capture Pack",
                f"Page ID: {page_id}",
                *content_lines,
                footer,
            ]
        )
    ) + "\n"


def _render_plain(
    page_id: str,
    lines: list[str],
    font_path: Path,
    dpi: int,
) -> tuple[Image.Image, str]:
    image, draw, body_font, meta_font, width, height, margin, y = _base(
        page_id,
        font_path,
        dpi,
    )
    body_width = width - 2 * margin
    line_spacing = round(dpi * 0.21)
    paragraph_spacing = round(dpi * 0.11)
    bottom_limit = height - margin - round(dpi * 0.4)

    rendered: list[str] = []
    for source_line in lines:
        wrapped = _wrap_text(draw, source_line, body_font, body_width)
        for wrapped_line in wrapped:
            if y + line_spacing > bottom_limit:
                break
            draw.text((margin, y), wrapped_line, font=body_font, fill="black")
            rendered.append(wrapped_line)
            y += line_spacing
        y += paragraph_spacing
        if y + line_spacing > bottom_limit:
            break

    footer = _footer(
        draw,
        page_id,
        meta_font,
        height=height,
        margin=margin,
    )
    return image, _truth(page_id, rendered, footer)


def _render_two_column(
    page_id: str,
    lines: list[str],
    font_path: Path,
    dpi: int,
) -> tuple[Image.Image, str]:
    image, draw, body_font, meta_font, width, height, margin, body_top = _base(
        page_id,
        font_path,
        dpi,
    )
    gap = round(dpi * 0.32)
    column_width = (width - 2 * margin - gap) // 2
    line_spacing = round(dpi * 0.20)
    paragraph_spacing = round(dpi * 0.10)
    bottom_limit = height - margin - round(dpi * 0.4)

    split = math.ceil(len(lines) / 2)
    columns = (lines[:split], lines[split:])
    rendered_columns: list[list[str]] = []

    for column_index, source_lines in enumerate(columns):
        x = margin + column_index * (column_width + gap)
        y = body_top
        rendered: list[str] = []
        for source_line in source_lines:
            for wrapped_line in _wrap_text(
                draw,
                source_line,
                body_font,
                column_width,
            ):
                if y + line_spacing > bottom_limit:
                    break
                draw.text((x, y), wrapped_line, font=body_font, fill="black")
                rendered.append(wrapped_line)
                y += line_spacing
            y += paragraph_spacing
            if y + line_spacing > bottom_limit:
                break
        rendered_columns.append(rendered)

    footer = _footer(
        draw,
        page_id,
        meta_font,
        height=height,
        margin=margin,
    )
    reading_order = [
        *rendered_columns[0],
        *rendered_columns[1],
    ]
    return image, _truth(page_id, reading_order, footer)


def _amount(row_index: int) -> str:
    return f"{(row_index + 1) * 5000:,} ₭"


def _render_table(
    page_id: str,
    lines: list[str],
    font_path: Path,
    dpi: int,
    *,
    ruled: bool,
) -> tuple[Image.Image, str]:
    image, draw, body_font, meta_font, width, height, margin, body_top = _base(
        page_id,
        font_path,
        dpi,
    )
    left = margin
    right = width - margin
    split_x = left + round((right - left) * 0.72)
    row_height = round(dpi * 0.48)
    rows = lines[: max(1, min(len(lines), 8))]
    table_rows = [("ລາຍການ", "ຈຳນວນ (₭)")]
    table_rows.extend(
        (text, _amount(index))
        for index, text in enumerate(rows)
    )

    if ruled:
        table_bottom = body_top + row_height * len(table_rows)
        for x in (left, split_x, right):
            draw.line((x, body_top, x, table_bottom), fill="black", width=max(1, dpi // 100))
        for row_index in range(len(table_rows) + 1):
            y = body_top + row_index * row_height
            draw.line((left, y, right, y), fill="black", width=max(1, dpi // 100))

    truth_rows: list[str] = []
    for row_index, (description, amount) in enumerate(table_rows):
        y = body_top + row_index * row_height + round(row_height * 0.22)
        description_font = _fit_font(
            draw,
            description,
            font_path,
            start_size=body_font.size,
            min_size=max(12, round(body_font.size * 0.70)),
            width=(split_x - left) - round(dpi * 0.12),
        )
        amount_font = _fit_font(
            draw,
            amount,
            font_path,
            start_size=body_font.size,
            min_size=max(12, round(body_font.size * 0.70)),
            width=(right - split_x) - round(dpi * 0.12),
        )
        draw.text(
            (left + round(dpi * 0.06), y),
            description,
            font=description_font,
            fill="black",
        )
        draw.text(
            (split_x + round(dpi * 0.06), y),
            amount,
            font=amount_font,
            fill="black",
        )
        truth_rows.append(f"{description}	{amount}")

    footer = _footer(
        draw,
        page_id,
        meta_font,
        height=height,
        margin=margin,
    )
    return image, _truth(page_id, truth_rows, footer)


def _render_receipt(
    page_id: str,
    lines: list[str],
    font_path: Path,
    dpi: int,
) -> tuple[Image.Image, str]:
    image, draw, body_font, meta_font, width, height, margin, y = _base(
        page_id,
        font_path,
        dpi,
    )
    title = "ໃບຮັບເງິນ / RECEIPT"
    title_font = ImageFont.truetype(
        str(font_path),
        size=max(body_font.size + 4, round(dpi * 0.14)),
    )
    draw.text((width // 2, y), title, font=title_font, fill="black", anchor="ma")
    y += round(dpi * 0.38)

    left = margin + round(dpi * 0.25)
    right = width - margin - round(dpi * 0.25)
    split_x = left + round((right - left) * 0.68)
    row_height = round(dpi * 0.34)
    items = lines[: max(1, min(len(lines), 9))]
    truth_rows = [title]
    total = 0

    for index, item in enumerate(items):
        amount_value = (index + 1) * 5000
        total += amount_value
        amount = f"{amount_value:,} ₭"
        item_font = _fit_font(
            draw,
            item,
            font_path,
            start_size=body_font.size,
            min_size=max(12, round(body_font.size * 0.70)),
            width=(split_x - left) - round(dpi * 0.10),
        )
        draw.text((left, y), item, font=item_font, fill="black")
        draw.text((right, y), amount, font=body_font, fill="black", anchor="ra")
        truth_rows.append(f"{item}	{amount}")
        y += row_height

    y += round(dpi * 0.12)
    draw.line((left, y, right, y), fill="black", width=max(1, dpi // 120))
    y += round(dpi * 0.10)
    total_text = f"TOTAL	{total:,} ₭"
    draw.text((left, y), "TOTAL", font=body_font, fill="black")
    draw.text((right, y), f"{total:,} ₭", font=body_font, fill="black", anchor="ra")
    truth_rows.append(total_text)

    footer = _footer(
        draw,
        page_id,
        meta_font,
        height=height,
        margin=margin,
    )
    return image, _truth(page_id, truth_rows, footer)


def _render_form(
    page_id: str,
    lines: list[str],
    font_path: Path,
    dpi: int,
) -> tuple[Image.Image, str]:
    image, draw, body_font, meta_font, width, height, margin, y = _base(
        page_id,
        font_path,
        dpi,
    )
    title = "ແບບຟອມ / FORM"
    title_font = ImageFont.truetype(
        str(font_path),
        size=max(body_font.size + 3, round(dpi * 0.14)),
    )
    draw.text((margin, y), title, font=title_font, fill="black")
    y += round(dpi * 0.40)

    label_width = round((width - 2 * margin) * 0.25)
    value_x = margin + label_width + round(dpi * 0.18)
    value_width = width - margin - value_x
    row_height = round(dpi * 0.50)
    values = lines[: max(1, min(len(lines), 9))]
    truth_rows = [title]

    for index, value in enumerate(values, start=1):
        label = f"Field {index}"
        fitted = _fit_font(
            draw,
            value,
            font_path,
            start_size=body_font.size,
            min_size=max(12, round(body_font.size * 0.68)),
            width=value_width,
        )
        draw.text((margin, y), label, font=body_font, fill="black")
        draw.text((value_x, y), value, font=fitted, fill="black")
        line_y = y + round(row_height * 0.60)
        draw.line(
            (value_x, line_y, width - margin, line_y),
            fill=(120, 120, 120),
            width=max(1, dpi // 150),
        )
        truth_rows.append(f"{label}	{value}")
        y += row_height

    footer = _footer(
        draw,
        page_id,
        meta_font,
        height=height,
        margin=margin,
    )
    return image, _truth(page_id, truth_rows, footer)


def render_capture_page(
    *,
    page_id: str,
    lines: list[str],
    font_path: str | Path,
    dpi: int,
    template: CaptureTemplate,
) -> tuple[Image.Image, str, tuple[str, ...]]:
    font = Path(font_path)
    if template == CaptureTemplate.PLAIN:
        image, truth = _render_plain(page_id, lines, font, dpi)
    elif template == CaptureTemplate.TWO_COLUMN:
        image, truth = _render_two_column(page_id, lines, font, dpi)
    elif template == CaptureTemplate.RULED_TABLE:
        image, truth = _render_table(
            page_id,
            lines,
            font,
            dpi,
            ruled=True,
        )
    elif template == CaptureTemplate.BORDERLESS_TABLE:
        image, truth = _render_table(
            page_id,
            lines,
            font,
            dpi,
            ruled=False,
        )
    elif template == CaptureTemplate.RECEIPT:
        image, truth = _render_receipt(page_id, lines, font, dpi)
    elif template == CaptureTemplate.FORM:
        image, truth = _render_form(page_id, lines, font, dpi)
    else:
        raise ValueError(f"Unsupported capture template: {template}")
    return image, truth, template_tags(template, lines)
