from __future__ import annotations

import html
import math
from functools import lru_cache
from pathlib import Path

import pymupdf
from PIL import Image


def missing_font_codepoints(
    font_path: str | Path,
    texts: list[str] | tuple[str, ...],
) -> tuple[int, ...]:
    font = pymupdf.Font(fontfile=str(font_path))
    codepoints = {
        ord(char)
        for text in texts
        for char in text
        if char not in {"\n", "\r", "\t"}
    }
    return tuple(sorted(cp for cp in codepoints if font.has_glyph(cp) == 0))


def validate_font_coverage(
    font_path: str | Path,
    texts: list[str] | tuple[str, ...],
    *,
    label: str = "font",
) -> None:
    missing = missing_font_codepoints(font_path, texts)
    if not missing:
        return

    preview = ", ".join(
        f"U+{codepoint:04X} {chr(codepoint)!r}"
        for codepoint in missing[:20]
    )
    remainder = len(missing) - min(len(missing), 20)
    suffix = f", plus {remainder} more" if remainder else ""
    raise ValueError(
        f"{label} is missing {len(missing)} required character(s): "
        f"{preview}{suffix}"
    )


@lru_cache(maxsize=4096)
def _shaped_rgba_bytes(
    font_path: str,
    font_size: int,
    text: str,
) -> tuple[int, int, bytes]:
    if not text:
        return 1, 1, bytes((0, 0, 0, 0))

    path = Path(font_path)
    if not path.is_file():
        raise FileNotFoundError(f"Font not found: {path}")
    if font_size < 1:
        raise ValueError("font_size must be positive")

    padding = max(8, font_size // 2)
    width = max(
        512,
        int(font_size * max(8, len(text)) * 2.0) + padding * 2,
    )
    height = max(64, font_size * 3 + padding * 2)

    document = pymupdf.open()
    try:
        page = document.new_page(width=width, height=height)
        archive = pymupdf.Archive(str(path.parent))
        css = (
            f"@font-face{{font-family:capturefont;src:url('{path.name}');}}"
            f"*{{font-family:capturefont;font-size:{font_size}pt;"
            "line-height:1;margin:0;padding:0;color:#000;}"
            "span{white-space:pre;}"
        )
        page.insert_htmlbox(
            pymupdf.Rect(
                padding,
                padding,
                width - padding,
                height - padding,
            ),
            f"<span>{html.escape(text)}</span>",
            css=css,
            archive=archive,
            scale_low=1,
        )
        pixmap = page.get_pixmap(alpha=True)
        rendered = Image.frombytes(
            "RGBA",
            (pixmap.width, pixmap.height),
            pixmap.samples,
        )
        bbox = rendered.getchannel("A").getbbox()
        if bbox is None:
            if text.isspace():
                font = pymupdf.Font(fontfile=str(path))
                advance = max(
                    1,
                    int(math.ceil(font.text_length(text, fontsize=font_size))),
                )
                transparent = Image.new("RGBA", (advance, 1), (0, 0, 0, 0))
                return advance, 1, transparent.tobytes()
            raise ValueError(f"Shaped text rendered no visible glyphs: {text!r}")
        cropped = rendered.crop(bbox)
        return cropped.width, cropped.height, cropped.tobytes()
    finally:
        document.close()


def shaped_text_image(
    font_path: str | Path,
    font_size: int,
    text: str,
) -> Image.Image:
    width, height, rgba = _shaped_rgba_bytes(
        str(Path(font_path).resolve()),
        int(font_size),
        text,
    )
    return Image.frombytes("RGBA", (width, height), rgba)


def paste_shaped_text(
    canvas: Image.Image,
    xy: tuple[float | int, float | int],
    text: str,
    *,
    font_path: str | Path,
    font_size: int,
    anchor: str | None = None,
) -> None:
    rendered = shaped_text_image(font_path, font_size, text)
    x = float(xy[0])
    y = float(xy[1])
    resolved_anchor = anchor or "la"

    horizontal = resolved_anchor[0]
    if horizontal == "m":
        x -= rendered.width / 2
    elif horizontal == "r":
        x -= rendered.width
    elif horizontal != "l":
        raise ValueError(f"Unsupported horizontal text anchor: {anchor!r}")

    vertical = resolved_anchor[1] if len(resolved_anchor) > 1 else "a"
    if vertical == "m":
        y -= rendered.height / 2
    elif vertical in {"b", "s"}:
        y -= rendered.height
    elif vertical not in {"a", "t"}:
        raise ValueError(f"Unsupported vertical text anchor: {anchor!r}")

    canvas.paste(rendered, (round(x), round(y)), rendered)
