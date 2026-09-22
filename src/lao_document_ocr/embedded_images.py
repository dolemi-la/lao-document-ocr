from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from PIL import Image

from lao_document_ocr.models import Block, BlockType, BoundingBox


@dataclass(frozen=True)
class EmbeddedImageAsset:
    bbox: BoundingBox
    png_bytes: bytes
    width_ratio: float
    xref: int

    def to_block(self) -> Block:
        return Block(
            type=BlockType.IMAGE,
            bbox=self.bbox,
            metadata={
                "source": "pdf-embedded",
                "media_type": "image/png",
                "image_base64": base64.b64encode(self.png_bytes).decode("ascii"),
                "width_ratio": self.width_ratio,
                "xref": self.xref,
            },
        )


def _normalize_to_png(data: bytes) -> bytes | None:
    try:
        with Image.open(io.BytesIO(data)) as source:
            if "A" in source.getbands():
                image = source.convert("RGBA")
            else:
                image = source.convert("RGB")
            output = io.BytesIO()
            image.save(output, format="PNG", optimize=True)
            return output.getvalue()
    except Exception:
        return None


def extract_pdf_embedded_images(
    pdf,
    page,
    *,
    rendered_width: int,
    rendered_height: int,
    min_area_ratio: float = 0.002,
    max_area_ratio: float = 0.75,
    max_source_pixels: int = 40_000_000,
) -> list[EmbeddedImageAsset]:
    page_rect = page.rect
    page_area = max(1.0, float(page_rect.width * page_rect.height))
    scale_x = rendered_width / max(1.0, float(page_rect.width))
    scale_y = rendered_height / max(1.0, float(page_rect.height))

    assets: list[EmbeddedImageAsset] = []
    seen: set[tuple[int, int, int, int, int]] = set()

    for image_info in page.get_images(full=True):
        xref = int(image_info[0])
        source_width = int(image_info[2]) if len(image_info) > 3 else 0
        source_height = int(image_info[3]) if len(image_info) > 3 else 0
        if (
            source_width > 0
            and source_height > 0
            and source_width * source_height > max_source_pixels
        ):
            continue
        try:
            extracted = pdf.extract_image(xref)
            raw = extracted.get("image")
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        if not isinstance(raw, (bytes, bytearray)):
            continue

        png_bytes = _normalize_to_png(bytes(raw))
        if png_bytes is None:
            continue

        for rect in rects:
            area_ratio = float(rect.width * rect.height) / page_area
            if area_ratio < min_area_ratio or area_ratio >= max_area_ratio:
                continue

            left = max(0, round(rect.x0 * scale_x))
            top = max(0, round(rect.y0 * scale_y))
            right = min(rendered_width, round(rect.x1 * scale_x))
            bottom = min(rendered_height, round(rect.y1 * scale_y))
            if right <= left or bottom <= top:
                continue

            key = (xref, left, top, right, bottom)
            if key in seen:
                continue
            seen.add(key)

            assets.append(
                EmbeddedImageAsset(
                    bbox=BoundingBox(
                        x=left,
                        y=top,
                        width=right - left,
                        height=bottom - top,
                    ),
                    png_bytes=png_bytes,
                    width_ratio=max(0.0, min(1.0, float(rect.width / page_rect.width))),
                    xref=xref,
                )
            )

    return sorted(assets, key=lambda asset: (asset.bbox.y, asset.bbox.x))
