"""Retained sideways output must be reviewable without serializing OCR content."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image

from lao_document_ocr.orientation_geometry import orientation_line_geometry
from lao_document_ocr.orientation_review import build_orientation_review
from lao_document_ocr.pipeline import _recognize_with_right_angle_orientation, process_document


def geometry(*, vertical=False, empty=False):
    lines = (
        []
        if empty
        else [
            SimpleNamespace(
                text="PRIVATE DOCUMENT TEXT",
                bbox=SimpleNamespace(width=20 if vertical else 100, height=100 if vertical else 20),
            )
            for _ in range(2)
        ]
    )
    return orientation_line_geometry(lines)


def page(number, *, selected=0, vertical=False, empty=False):
    return {
        "page": number,
        "degrees_clockwise": selected,
        "diagnostics": {
            "candidate_geometry": [
                {
                    "degrees_clockwise": selected,
                    "geometry": geometry(vertical=vertical, empty=empty),
                },
            ],
            "private": "PRIVATE DIAGNOSTIC TEXT",
        },
    }


def test_review_uses_selected_geometry_not_rejected_candidate():
    first = page(1, selected=90)
    first["diagnostics"]["candidate_geometry"].append(
        {"degrees_clockwise": 0, "geometry": geometry(vertical=True)}
    )
    metadata = {"enabled": True, "pages": [first, page(2, vertical=True), page(3, empty=True)]}
    before = deepcopy(metadata)
    report = build_orientation_review(metadata, page_count=3).to_dict()
    assert report == {
        "version": "selected-line-axis-review-v1",
        "status": "review-required",
        "page_count": 3,
        "review_pages": [2],
        "unassessed_pages": [3],
    }
    assert metadata == before
    assert "PRIVATE" not in json.dumps(report)


def test_all_assessed_is_no_sideways_evidence_not_verified_upright():
    report = build_orientation_review(
        {"enabled": True, "pages": [page(1), page(2, selected=180)]},
        page_count=2,
    ).to_dict()
    assert report["status"] == "no-sideways-evidence"
    assert report["review_pages"] == report["unassessed_pages"] == []


@pytest.mark.parametrize(
    "metadata,status",
    [
        ({"enabled": False}, "not-requested"),
        (None, "not-assessed"),
        ({}, "not-assessed"),
        ({"enabled": "true"}, "not-assessed"),
        ({"enabled": True}, "incomplete"),
    ],
)
def test_missing_or_disabled_assessment_is_never_invented(metadata, status):
    report = build_orientation_review(metadata, page_count=2).to_dict()
    assert report["status"] == status
    assert report["review_pages"] == []
    assert report["unassessed_pages"] == [1, 2]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda record: record.update(page=True),
        lambda record: record.update(degrees_clockwise="0"),
        lambda record: record.update(degrees_clockwise=45),
        lambda record: record.update(diagnostics=None),
        lambda record: record["diagnostics"].update(candidate_geometry=[]),
        lambda record: record["diagnostics"]["candidate_geometry"].append(
            {"degrees_clockwise": 0, "geometry": geometry()},
        ),
        lambda record: record["diagnostics"]["candidate_geometry"][0]["geometry"].update(
            version="unsupported",
        ),
        lambda record: record["diagnostics"]["candidate_geometry"][0]["geometry"].update(
            horizontal_characters="PRIVATE STRING",
        ),
    ],
)
def test_invalid_or_ambiguous_selected_evidence_is_unassessed(mutation):
    record = page(1)
    mutation(record)
    report = build_orientation_review({"enabled": True, "pages": [record]}, page_count=1).to_dict()
    assert report["status"] == "incomplete"
    assert report["review_pages"] == []
    assert report["unassessed_pages"] == [1]


def test_duplicate_page_entries_are_not_used_and_output_is_defensive():
    review = build_orientation_review(
        {"enabled": True, "pages": [page(1), page(1, vertical=True), page(2, vertical=True)]},
        page_count=2,
    )
    report = review.to_dict()
    assert report["review_pages"] == [2]
    assert report["unassessed_pages"] == [1]
    report["review_pages"].clear()
    assert review.to_dict()["review_pages"] == [2]


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_invalid_page_count_fails(count):
    with pytest.raises(ValueError, match="page_count"):
        build_orientation_review(None, page_count=count)


def test_empty_document_does_not_claim_completed_assessment():
    assert (
        build_orientation_review({"enabled": True, "pages": []}, page_count=0).to_dict()["status"]
        == "incomplete"
    )


def test_real_pipeline_retains_same_image_and_lines_but_flags_review(tmp_path):
    from lao_document_ocr.models import BoundingBox
    from lao_document_ocr.ocr.base import OcrEngine, RecognizedLine

    class EqualConfidenceEngine(OcrEngine):
        def __init__(self):
            self.calls = 0

        def is_available(self):
            return True

        def recognize(self, image):
            self.calls += 1
            vertical = image.height > image.width
            return [
                RecognizedLine(
                    "project authored equal confidence fixture",
                    BoundingBox(
                        x=20 + index * 30,
                        y=20,
                        width=20 if vertical else 150,
                        height=150 if vertical else 20,
                    ),
                    0.85,
                    1,
                    1,
                    index,
                )
                for index in (1, 2)
            ]

    source = tmp_path / "fixture.png"
    Image.new("RGB", (400, 600), "white").save(source)
    engine = EqualConfidenceEngine()
    document = process_document(source, engine=engine, auto_orient_right_angles=True)
    review = document.metadata["auto_orientation"]["review"]
    assert engine.calls == 4
    assert document.metadata["auto_orientation"]["pages"][0]["degrees_clockwise"] == 0
    assert (document.pages[0].width, document.pages[0].height) == (400, 600)
    assert review["status"] == "review-required"
    assert review["review_pages"] == [1]
    assert "fixture" not in json.dumps(review)

    default_engine = EqualConfidenceEngine()
    default = process_document(source, engine=default_engine)
    assert default_engine.calls == 1
    assert default.metadata["auto_orientation"]["review"]["status"] == "not-requested"
    assert default.pages == document.pages

    # The recognition helper still returns the exact unrotated input on a tie.
    image = Image.new("RGB", (400, 600), "white")
    selected, _, angle, _ = _recognize_with_right_angle_orientation(
        image,
        engine=EqualConfidenceEngine(),
    )
    assert selected is image and angle == 0


@pytest.mark.parametrize(
    "arguments",
    [
        {"page_count": 2, "requested": "true", "review_pages": (), "unassessed_pages": ()},
        {"page_count": 2, "requested": True, "review_pages": [1], "unassessed_pages": ()},
        {"page_count": 2, "requested": True, "review_pages": (True,), "unassessed_pages": ()},
        {"page_count": 2, "requested": True, "review_pages": (3,), "unassessed_pages": ()},
        {"page_count": 2, "requested": True, "review_pages": (1, 1), "unassessed_pages": ()},
        {"page_count": 2, "requested": True, "review_pages": (2, 1), "unassessed_pages": ()},
        {"page_count": 2, "requested": True, "review_pages": (1,), "unassessed_pages": (1,)},
        {"page_count": 2, "requested": False, "review_pages": (1,), "unassessed_pages": (2,)},
        {"page_count": 2, "requested": None, "review_pages": (), "unassessed_pages": ()},
    ],
)
def test_typed_summary_rejects_invalid_or_fabricated_page_sets(arguments):
    from lao_document_ocr.orientation_review import OrientationReview

    with pytest.raises(ValueError):
        OrientationReview(**arguments)


def test_review_recomputes_untrusted_sideways_flag_and_ignores_inconsistent_counts():
    record = page(1)
    data = record["diagnostics"]["candidate_geometry"][0]["geometry"]
    data["sideways_dominant"] = True
    assert (
        build_orientation_review({"enabled": True, "pages": [record]}, page_count=1).to_dict()[
            "status"
        ]
        == "no-sideways-evidence"
    )
    data["horizontal_lines"] = 0
    assert (
        build_orientation_review({"enabled": True, "pages": [record]}, page_count=1).to_dict()[
            "status"
        ]
        == "incomplete"
    )
