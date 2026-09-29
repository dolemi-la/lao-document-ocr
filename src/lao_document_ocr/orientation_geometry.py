"""No-text line-axis evidence for horizontal Lao/English page orientation.

This is a veto for proposed right-angle candidates, not a reading-direction
classifier. A recognizer may decode sideways text confidently while returning
its boxes in the input image coordinates. Box evidence does not distinguish
upright from upside-down text, and uncertain geometry must remain unknown.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from lao_document_ocr.ocr.base import RecognizedLine

ORIENTATION_GEOMETRY_VERSION = "multi-character-line-axis-v1"


def orientation_line_geometry(lines: Iterable[RecognizedLine]) -> dict[str, Any]:
    """Count multi-character, non-square line axes without retaining text/boxes.

    One-character labels, square boxes and invalid/nonpositive dimensions do not
    determine an axis. A veto requires at least two vertical lines and strictly
    more character support in vertical than horizontal lines. This intentionally
    ignores OCR confidence and supplies no guarantee of transcription accuracy.
    """
    counts = {
        "horizontal_lines": 0,
        "vertical_lines": 0,
        "horizontal_characters": 0,
        "vertical_characters": 0,
        "ignored_lines": 0,
    }
    for line in lines:
        characters = sum(not char.isspace() for char in line.text)
        width, height = float(line.bbox.width), float(line.bbox.height)
        if (
            characters < 2
            or not math.isfinite(width)
            or not math.isfinite(height)
            or min(width, height) <= 0
            or width == height
        ):
            counts["ignored_lines"] += 1
            continue
        axis = "horizontal" if width > height else "vertical"
        counts[f"{axis}_lines"] += 1
        counts[f"{axis}_characters"] += characters
    return {
        "version": ORIENTATION_GEOMETRY_VERSION,
        **counts,
        "sideways_dominant": (
            counts["vertical_lines"] >= 2
            and counts["vertical_characters"] > counts["horizontal_characters"]
        ),
    }


def sanitized_orientation_geometry(payload: object) -> dict[str, Any] | None:
    """Accept only known numeric geometry metadata, never arbitrary report keys."""
    if not isinstance(payload, dict) or payload.get("version") != ORIENTATION_GEOMETRY_VERSION:
        return None
    keys = (
        "horizontal_lines",
        "vertical_lines",
        "horizontal_characters",
        "vertical_characters",
        "ignored_lines",
    )
    if any(type(payload.get(key)) is not int or payload[key] < 0 for key in keys):
        return None
    counts = {key: payload[key] for key in keys}
    return {
        "version": ORIENTATION_GEOMETRY_VERSION,
        **counts,
        "sideways_dominant": (
            counts["vertical_lines"] >= 2
            and counts["vertical_characters"] > counts["horizontal_characters"]
        ),
    }
