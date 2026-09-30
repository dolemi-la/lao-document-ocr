"""No-text review signals for the selected output, never orientation certification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lao_document_ocr.orientation_geometry import sanitized_orientation_geometry

ORIENTATION_REVIEW_VERSION = "selected-line-axis-review-v1"
_ANGLES = {0, 90, 180, 270}


@dataclass(frozen=True)
class OrientationReview:
    page_count: int
    requested: bool | None
    review_pages: tuple[int, ...]
    unassessed_pages: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.page_count) is not int or self.page_count < 0:
            raise ValueError("page_count must be a non-negative integer")
        if self.requested is not None and type(self.requested) is not bool:
            raise ValueError("requested must be a boolean or None")
        for pages in (self.review_pages, self.unassessed_pages):
            if not isinstance(pages, tuple) or any(
                type(page) is not int or not 1 <= page <= self.page_count for page in pages
            ):
                raise ValueError("Review page numbers must be bounded integer tuples")
            if tuple(sorted(set(pages))) != pages:
                raise ValueError("Review page numbers must be sorted and unique")
        if set(self.review_pages) & set(self.unassessed_pages):
            raise ValueError("Reviewed and unassessed page sets must be disjoint")
        if self.requested is not True and (
            self.review_pages or self.unassessed_pages != tuple(range(1, self.page_count + 1))
        ):
            raise ValueError("Unrequested or unknown assessments cannot classify pages")

    def to_dict(self) -> dict[str, Any]:
        if self.requested is False:
            status = "not-requested"
        elif self.requested is None:
            status = "not-assessed"
        elif self.review_pages:
            status = "review-required"
        elif self.unassessed_pages or self.page_count == 0:
            status = "incomplete"
        else:
            # No flag is NOT proof of upright text, correct OCR, or complete coverage.
            status = "no-sideways-evidence"
        return {
            "version": ORIENTATION_REVIEW_VERSION,
            "status": status,
            "page_count": self.page_count,
            "review_pages": list(self.review_pages),
            "unassessed_pages": list(self.unassessed_pages),
        }


def _selected_geometry(record: dict) -> dict | None:
    angle = record.get("degrees_clockwise")
    if type(angle) is not int or angle not in _ANGLES:
        return None
    diagnostics = record.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    candidates = diagnostics.get("candidate_geometry")
    if not isinstance(candidates, list):
        return None
    matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and type(candidate.get("degrees_clockwise")) is int
        and candidate["degrees_clockwise"] == angle
    ]
    if len(matches) != 1:
        return None
    geometry = sanitized_orientation_geometry(matches[0].get("geometry"))
    if geometry is None or geometry["horizontal_lines"] + geometry["vertical_lines"] == 0:
        return None
    # Inconsistent external/legacy metadata must not invent assessable evidence.
    for axis in ("horizontal", "vertical"):
        lines, characters = geometry[f"{axis}_lines"], geometry[f"{axis}_characters"]
        if (lines == 0 and characters != 0) or characters < 2 * lines:
            return None
    return geometry


def build_orientation_review(payload: object, *, page_count: int) -> OrientationReview:
    """Summarize only selected geometry; ignore rejected candidates and arbitrary text.

    Missing, duplicate, unknown-version, or non-informative evidence stays
    unassessed. These are heuristic review flags, not ground-truth annotations.
    No OCR, rotation, threshold change, or source-specific rule is performed.
    """
    if type(page_count) is not int or page_count < 0:
        raise ValueError("page_count must be a non-negative integer")
    enabled = payload.get("enabled") if isinstance(payload, dict) else None
    requested = enabled if type(enabled) is bool else None
    if requested is not True:
        return OrientationReview(page_count, requested, (), tuple(range(1, page_count + 1)))

    records: dict[int, list[dict]] = {}
    pages = payload.get("pages")
    for record in pages if isinstance(pages, list) else []:
        if not isinstance(record, dict):
            continue
        number = record.get("page")
        if type(number) is int and 1 <= number <= page_count:
            records.setdefault(number, []).append(record)

    review_pages: list[int] = []
    unassessed_pages: list[int] = []
    for number in range(1, page_count + 1):
        matches = records.get(number, [])
        geometry = _selected_geometry(matches[0]) if len(matches) == 1 else None
        if geometry is None:
            unassessed_pages.append(number)
        elif geometry["sideways_dominant"]:
            review_pages.append(number)
    return OrientationReview(page_count, True, tuple(review_pages), tuple(unassessed_pages))
