"""Manual corrections survive HTTP validation, real workers and export packaging."""

from __future__ import annotations

import io
import json
import time
import zipfile

import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import services.api.app.main as api
from lao_document_ocr.models import BoundingBox
from lao_document_ocr.ocr.base import OcrEngine, RecognizedLine
from lao_document_ocr.orientation_review import build_orientation_review
from services.api.app.jobs import ConversionJobManager, JobStatus
from services.api.app.storage import FilesystemArtifactStorage

ENDPOINTS = ("/v1/parse", "/v1/convert", "/v1/jobs", "/v1/jobs/batch")


class RecordingEngine(OcrEngine):
    def __init__(self):
        self.calls = []

    def is_available(self):
        return True

    def recognize(self, image):
        self.calls.append(image.size)
        vertical = image.height > image.width
        return [
            RecognizedLine(
                "PRIVATE authored rotation fixture",
                BoundingBox(
                    x=30 + i * 30,
                    y=20,
                    width=20 if vertical else 180,
                    height=180 if vertical else 20,
                ),
                0.85,
                1,
                1,
                i,
            )
            for i in (1, 2)
        ]

    def metadata(self):
        return {"name": "fixture", "secret": "PRIVATE ENGINE DATA"}


def png():
    output = io.BytesIO()
    Image.new("RGB", (400, 600), "white").save(output, format="PNG")
    return output.getvalue()


def pdf(pages):
    with pymupdf.open() as document:
        for _ in range(pages):
            document.new_page(width=200, height=300)
        return document.tobytes()


@pytest.fixture
def context(tmp_path, monkeypatch):
    engine = RecordingEngine()
    storage = FilesystemArtifactStorage(tmp_path / "artifacts")
    monkeypatch.setattr(api, "_engine", lambda: engine)
    monkeypatch.setattr(api, "_enforce_submission_rate_limit", lambda *a, **kw: None)
    monkeypatch.setattr(api, "RESULT_STORAGE", storage)
    manager = ConversionJobManager(
        tmp_path / "jobs",
        api._run_conversion_job,
        max_workers=1,
        max_active_jobs=4,
        artifact_exists=storage.exists,
        artifact_cleanup=storage.delete,
    )
    monkeypatch.setattr(api, "JOB_MANAGER", manager)
    try:
        with TestClient(api.app) as client:
            yield client, manager, engine, tmp_path
    finally:
        manager.shutdown()


def wait(client, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            assert job["status"] == "succeeded", job
            return job
        assert job["orientation_review"] is None
        time.sleep(0.01)
    raise AssertionError("Job did not complete")


def archive_document(content):
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        assert {name.rsplit(".", 1)[-1] for name in archive.namelist()} == {
            "json",
            "docx",
            "md",
            "txt",
        }
        return json.loads(archive.read(next(n for n in archive.namelist() if n.endswith(".json"))))


def submit(client, endpoint, data=None, uploads=None):
    file_field = "files" if endpoint.endswith("batch") else "file"
    uploads = uploads or [("page.png", png(), "image/png")]
    return client.post(endpoint, files=[(file_field, value) for value in uploads], data=data or {})


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("angle", [0, 90, 180, 270])
@pytest.mark.parametrize("auto", [False, True])
def test_manual_form_actual_pipeline_and_review(context, endpoint, angle, auto):
    client, _, engine, _ = context
    response = submit(
        client,
        endpoint,
        {
            "rotate_page": [f"1:{angle}"],
            "auto_orient_right_angles": str(auto).lower(),
        },
    )
    assert response.status_code == (202 if "/jobs" in endpoint else 200)
    job = None
    if "/jobs" in endpoint:
        created = response.json()
        if endpoint.endswith("batch"):
            created = created["jobs"][0]
        expected = [{"page": 1, "degrees_clockwise": angle}]
        assert created["page_rotations"] == expected
        job = wait(client, created["id"])
        assert job["page_rotations"] == expected
        assert job["download_ready"]
        assert "PRIVATE" not in json.dumps(job)
        download = client.get(f"/v1/jobs/{job['id']}/download")
        assert download.status_code == 200
        document = archive_document(download.content)
    elif endpoint.endswith("parse"):
        document = response.json()
    else:
        document = archive_document(response.content)
    width, height = (600, 400) if angle in (90, 270) else (400, 600)
    assert engine.calls == [(width, height)]  # Even explicit 0 overrides auto probes.
    assert (document["pages"][0]["width"], document["pages"][0]["height"]) == (width, height)
    manual = document["metadata"]["manual_page_rotations"]
    assert manual["pages"] == [{"page": 1, "degrees_clockwise": angle}]
    assert manual["review"]["review_pages"] == ([] if width > height else [1])
    if job:
        assert job["orientation_review"] == manual["review"]
    assert document["metadata"]["auto_orientation"]["enabled"] is auto


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(
    "specs",
    [
        ["0:90"],
        ["1:45"],
        ["-1:90"],
        ["01:90"],
        ["1:360"],
        ["1:-90"],
        ["true:90"],
        ["1:90.0"],
        ["1:90,2:270"],
        ["1:90", "1:90"],
        ["1:0", "1:270"],
        [""],
        ["1:90 "],
        ["2:90"],
        ["9" * 40 + ":90"],
    ],
)
def test_invalid_form_is_rejected_without_ocr_or_reserved_jobs(context, endpoint, specs):
    client, manager, engine, root = context
    response = submit(client, endpoint, {"rotate_page": specs})
    assert response.status_code == 422, response.text
    assert not engine.calls
    assert not manager._jobs
    assert not list((root / "jobs").iterdir())
    assert not list((root / "artifacts").rglob("*.zip"))


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_omitted_form_retains_default_output(context, endpoint):
    client, _, engine, _ = context
    response = submit(client, endpoint)
    assert response.status_code == (202 if "/jobs" in endpoint else 200)
    if "/jobs" in endpoint:
        created = response.json()
        created = created["jobs"][0] if endpoint.endswith("batch") else created
        job = wait(client, created["id"])
        assert job["page_rotations"] == []
        assert job["orientation_review"]["status"] == "not-requested"
    else:
        document = (
            response.json() if endpoint.endswith("parse") else archive_document(response.content)
        )
        assert "manual_page_rotations" not in document["metadata"]
    assert engine.calls == [(400, 600)]


def test_batch_page_bounds_reject_whole_batch_before_enqueue(context):
    client, manager, engine, root = context
    response = submit(
        client,
        "/v1/jobs/batch",
        {"rotate_page": ["2:90"]},
        [
            ("two.pdf", pdf(2), "application/pdf"),
            ("one.png", png(), "image/png"),
        ],
    )
    assert response.status_code == 422
    assert "page count (1)" in response.json()["detail"]
    assert not engine.calls and not manager._jobs
    assert not list((root / "jobs").iterdir())
    assert not list((root / "artifacts").rglob("*.zip"))


def test_batch_same_mapping_is_local_to_each_document(context):
    client, _, engine, _ = context
    response = submit(
        client,
        "/v1/jobs/batch",
        {"rotate_page": ["2:90", "1:0"]},
        [
            ("two.pdf", pdf(2), "application/pdf"),
            ("three.pdf", pdf(3), "application/pdf"),
        ],
    )
    assert response.status_code == 202
    for created, expected_unknown in zip(response.json()["jobs"], ([], [3]), strict=True):
        job = wait(client, created["id"])
        assert job["page_rotations"] == [
            {"page": 1, "degrees_clockwise": 0},
            {"page": 2, "degrees_clockwise": 90},
        ]
        assert job["orientation_review"]["review_pages"] == [1]
        assert job["orientation_review"]["unassessed_pages"] == expected_unknown
        doc = archive_document(client.get(f"/v1/jobs/{job['id']}/download").content)
        assert doc["metadata"]["manual_page_rotations"]["review"] == job["orientation_review"]
    assert len(engine.calls) == 5


def test_form_maximum_is_bounded_before_reservation(context, monkeypatch):
    client, manager, engine, _ = context
    monkeypatch.setattr(api, "MAX_PAGES", 2)
    for specs in (["1:90", "2:0", "3:180"], ["3:90"]):
        response = submit(client, "/v1/jobs", {"rotate_page": specs})
        assert response.status_code == 422
    assert not manager._jobs and not engine.calls


def test_job_rotations_are_independent_immutable_snapshots(context):
    _, manager, _, _ = context
    source = {2: 270, 1: 0}
    first, second = manager.reserve_many(
        [("a.pdf", ".pdf"), ("b.pdf", ".pdf")], page_rotations=source
    )
    source[1] = 90
    assert dict(first.page_rotations) == dict(second.page_rotations) == {1: 0, 2: 270}
    with pytest.raises(TypeError):
        first.page_rotations[1] = 90
    exposed = manager.public(first.id)
    exposed["page_rotations"][0]["degrees_clockwise"] = 90
    assert manager.public(first.id)["page_rotations"][0]["degrees_clockwise"] == 0
    with pytest.raises(ValueError):
        manager.reserve("invalid.png", ".png", page_rotations={True: 90})
    assert len(manager._jobs) == 2


def test_openapi_advertises_optional_repeated_string_form(context):
    client, _, _, _ = context
    schema = client.get("/openapi.json").json()
    for endpoint in ENDPOINTS:
        ref = schema["paths"][endpoint]["post"]["requestBody"]["content"]["multipart/form-data"][
            "schema"
        ]
        body = schema["components"]["schemas"][ref["$ref"].rsplit("/", 1)[-1]]
        field = body["properties"]["rotate_page"]
        variants = field.get("anyOf", [field])
        assert any(
            item.get("type") == "array" and item["items"]["type"] == "string" for item in variants
        )
        assert "rotate_page" not in body.get("required", [])


def test_invalid_batch_does_not_discard_unrelated_job(context):
    client, manager, engine, _ = context
    previous = manager.reserve("existing.png", ".png", page_rotations={1: 0})
    previous.input_path.write_bytes(b"preserve-existing-upload")
    response = submit(
        client,
        "/v1/jobs/batch",
        {"rotate_page": ["2:90"]},
        [
            ("two.pdf", pdf(2), "application/pdf"),
            ("one.png", png(), "image/png"),
        ],
    )
    assert response.status_code == 422
    assert list(manager._jobs) == [previous.id]
    assert previous.input_path.read_bytes() == b"preserve-existing-upload"
    assert not engine.calls


@pytest.mark.parametrize("status", list(JobStatus))
def test_manual_job_only_exposes_completed_review_after_success(context, status):
    _, manager, _, _ = context
    record = manager.reserve("manual.png", ".png", page_rotations={1: 0})
    record.orientation_review = build_orientation_review(None, page_count=1)
    record.status = status
    public = manager.public(record.id)
    assert public["page_rotations"] == [{"page": 1, "degrees_clockwise": 0}]
    if status == JobStatus.SUCCEEDED:
        assert public["orientation_review"]["status"] == "not-assessed"
    else:
        assert public["orientation_review"] is None
