"""Explicit page corrections, separate from automatic orientation decisions."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

MANUAL_PAGE_ROTATION_VERSION = "explicit-page-rotation-v1"
_ANGLES = frozenset({0, 90, 180, 270})
_SPEC = re.compile(r"([1-9][0-9]*):(0|90|180|270)")


def validate_page_rotations(
    rotations: Mapping[int, int] | None,
    *,
    page_count: int | None = None,
) -> dict[int, int]:
    """Copy validated one-based page/clockwise-angle pairs without coercion.

    Zero is an explicit override, not a missing value. Bounds are checked after
    loading the document and before any OCR. No angle is silently normalized.
    """
    if page_count is not None and (type(page_count) is not int or page_count < 0):
        raise ValueError("page_count must be a non-negative integer")
    if rotations is None:
        return {}
    if not isinstance(rotations, Mapping):
        raise ValueError("page_rotations must map page integers to clockwise angle integers")
    result: dict[int, int] = {}
    for page, angle in rotations.items():
        if type(page) is not int or page < 1:
            raise ValueError("Rotation page numbers must be positive integers")
        if type(angle) is not int or angle not in _ANGLES:
            raise ValueError("Rotation angles must be integer 0, 90, 180, or 270 degrees clockwise")
        if page_count is not None and page > page_count:
            raise ValueError(f"Rotation page {page} exceeds the document page count ({page_count})")
        result[page] = angle
    return dict(sorted(result.items()))


def parse_page_rotations(specifications: Iterable[str]) -> dict[int, int]:
    """Parse repeated CLI PAGE:DEGREES values, rejecting even identical duplicates."""
    rotations: dict[int, int] = {}
    for specification in specifications:
        match = _SPEC.fullmatch(specification) if isinstance(specification, str) else None
        if match is None:
            raise ValueError(
                "--rotate-page requires PAGE:DEGREES, e.g. 2:90; angles: 0, 90, 180, 270"
            )
        page, angle = (int(part) for part in match.groups())
        if page in rotations:
            raise ValueError(f"--rotate-page repeats page {page}; specify each page only once")
        rotations[page] = angle
    return validate_page_rotations(rotations)
