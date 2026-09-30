"""Review flags survive real API workers/exports, not only mocked process results."""

from __future__ import annotations

import io
import json
import time
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import services.api.app.main as api
from lao_document_ocr.models import BoundingBox
from lao_document_ocr.ocr.base import OcrEngine, RecognizedLine
from lao_document_ocr.orientation_review import build_orientation_review
from services.api.app.jobs import ConversionJobManager, JobStatus
from services.api.app.storage import FilesystemArtifactStorage


class EqualConfidenceEngine(OcrEngine):
    def is_available(self):
        return True

    def recognize(self, image):
        vertical = image.height > image.width
        return [
            RecognizedLine(
                "PRIVATE fixture text for review tests",
                BoundingBox(
                    x=20 + i * 30,
                    y=20,
                    width=20 if vertical else 150,
                    height=150 if vertical else 20,
                ),
                0.85,
                1,
                1,
                i,
            )
            for i in (1, 2)
        ]

    def metadata(self):
        return {"name": "fake", "secret": "PRIVATE ENGINE METADATA"}


def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (400, 600), "white").save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    storage = FilesystemArtifactStorage(tmp_path / "artifacts")
    monkeypatch.setattr(api, "_engine", EqualConfidenceEngine)
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
        yield TestClient(api.app)
    finally:
        manager.shutdown()


def _wait(client, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"succeeded", "failed", "cancelled"}:
            assert payload["status"] == "succeeded", payload
            return payload
        assert payload["orientation_review"] is None
        time.sleep(0.01)
    raise AssertionError("Job did not complete")


def _archive_review(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        name = next(name for name in archive.namelist() if name.endswith(".json"))
        return json.loads(archive.read(name))["metadata"]["auto_orientation"]["review"]


@pytest.mark.parametrize("enabled", [False, True])
def test_parse_and_sync_archive_have_same_review(api_client, enabled):
    kwargs = {
        "files": {"file": ("sample.png", image_bytes(), "image/png")},
        "data": {"auto_orient_right_angles": str(enabled).lower()},
    }
    parsed = api_client.post("/v1/parse", **kwargs)
    assert parsed.status_code == 200
    review = parsed.json()["metadata"]["auto_orientation"]["review"]
    assert review["status"] == ("review-required" if enabled else "not-requested")
    assert review["review_pages"] == ([1] if enabled else [])
    assert "PRIVATE" not in json.dumps(review)
    archive = api_client.post("/v1/convert", **kwargs)
    assert archive.status_code == 200
    assert _archive_review(archive.content) == review


@pytest.mark.parametrize("enabled", [False, True])
def test_job_status_matches_actual_download_summary(api_client, enabled):
    created = api_client.post(
        "/v1/jobs",
        files={"file": ("sample.png", image_bytes(), "image/png")},
        data={"auto_orient_right_angles": str(enabled).lower()},
    )
    assert created.status_code == 202
    job = _wait(api_client, created.json()["id"])
    assert "PRIVATE" not in json.dumps(job)
    assert job["orientation_review"]["status"] == (
        "review-required" if enabled else "not-requested"
    )
    download = api_client.get(f"/v1/jobs/{job['id']}/download")
    assert download.status_code == 200
    assert _archive_review(download.content) == job["orientation_review"]
    assert job["download_ready"] is True


def test_batch_members_keep_independent_page_numbers(api_client):
    created = api_client.post(
        "/v1/jobs/batch",
        files=[("files", (name, image_bytes(), "image/png")) for name in ("a.png", "b.png")],
        data={"auto_orient_right_angles": "true"},
    )
    assert created.status_code == 202
    jobs = [_wait(api_client, job["id"]) for job in created.json()["jobs"]]
    assert len(jobs) == 2
    assert all(job["orientation_review"]["review_pages"] == [1] for job in jobs)
    assert "PRIVATE" not in json.dumps(jobs)


@pytest.mark.parametrize("status", list(JobStatus))
def test_job_summary_is_only_visible_after_success(tmp_path, status):
    manager = ConversionJobManager(tmp_path / "jobs", lambda *args: tmp_path / "result.zip")
    try:
        record = manager.reserve("sample.png", ".png")
        record.orientation_review = build_orientation_review(None, page_count=2)
        record.status = status
        summary = manager.public(record.id)["orientation_review"]
        if status == JobStatus.SUCCEEDED:
            assert summary["status"] == "not-assessed"
            summary["unassessed_pages"].clear()
            assert manager.public(record.id)["orientation_review"]["unassessed_pages"] == [1, 2]
        else:
            assert summary is None
    finally:
        manager.shutdown()


def test_legacy_runner_has_no_fabricated_review(tmp_path):
    manager = ConversionJobManager(tmp_path / "jobs", lambda *args: tmp_path / "result.zip")
    try:
        record = manager.reserve("sample.png", ".png")
        record.status = JobStatus.SUCCEEDED
        assert manager.public(record.id)["orientation_review"] is None
        record.orientation_review = {"status": "PRIVATE malicious metadata"}
        assert manager.public(record.id)["orientation_review"] is None
    finally:
        manager.shutdown()
