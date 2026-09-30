"""Durable cleanup stays opt-in and is wired to a persistent deployment root."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_storage import FakeS3Client

from services.api.app.cleanup_journal import JOURNAL_FILENAME
from services.api.app.storage import S3ArtifactStorage

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value, enabled", [("true", True), ("false", False)])
def test_api_constructs_the_configured_cleanup_mode(tmp_path, value, enabled):
    env = {
        **os.environ,
        "JOB_CLEANUP_DURABLE": value,
        "JOB_ROOT": str(tmp_path / "jobs"),
        "RESULT_STORAGE_BACKEND": "filesystem",
        "RESULT_STORAGE_ROOT": str(tmp_path / "objects"),
    }
    code = """
import sys
import services.api.app.main as api
try:
    assert api.JOB_CLEANUP_DURABLE is (sys.argv[1] == "true")
    assert api.JOB_MANAGER.durable_cleanup is api.JOB_CLEANUP_DURABLE
    assert (api.JOB_MANAGER._journal is not None) is api.JOB_CLEANUP_DURABLE
finally:
    api.JOB_MANAGER.shutdown()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, value],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "jobs" / JOURNAL_FILENAME).exists() is enabled


def test_invalid_mode_fails_before_allocating_job_storage(tmp_path):
    env = {
        **os.environ,
        "JOB_CLEANUP_DURABLE": "PRIVATE invalid setting",
        "JOB_ROOT": str(tmp_path / "jobs"),
        "RESULT_STORAGE_BACKEND": "filesystem",
        "RESULT_STORAGE_ROOT": str(tmp_path / "objects"),
    }
    result = subprocess.run(
        [sys.executable, "-c", "import services.api.app.main"],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode != 0
    assert "JOB_CLEANUP_DURABLE must be a boolean value" in result.stderr
    assert "PRIVATE" not in result.stderr
    assert not (tmp_path / "jobs").exists()
    assert not (tmp_path / "objects").exists()


def test_sdk_resolved_endpoint_and_region_are_bound_without_credentials():
    client = FakeS3Client()
    client.meta = SimpleNamespace(endpoint_url="https://resolved.invalid", region_name="fixture-1")
    storage = S3ArtifactStorage("fixture-bucket", prefix="/prefix/", client=client)
    identity = storage.cleanup_namespace()
    assert identity == {
        "backend": "s3",
        "bucket": "fixture-bucket",
        "prefix": "prefix",
        "endpoint_url": "https://resolved.invalid",
        "region": "fixture-1",
    }
    client.meta.endpoint_url = "https://different.invalid"
    assert storage.cleanup_namespace() != identity


def test_unresolved_s3_endpoint_cannot_enable_durable_cleanup():
    storage = S3ArtifactStorage("fixture-bucket", client=FakeS3Client())
    with pytest.raises(ValueError, match="resolved endpoint"):
        storage.cleanup_namespace()


def test_default_compose_and_presets_do_not_silently_enable_replay():
    compose = (ROOT / "docker-compose.yml").read_text()
    assert 'JOB_CLEANUP_DURABLE: "${JOB_CLEANUP_DURABLE:-false}"' in compose
    assert 'JOB_ROOT: "${JOB_ROOT:-/tmp/lao-document-ocr-jobs}"' in compose
    for path in (ROOT / "deploy/presets").glob("*.env.example"):
        assert "JOB_CLEANUP_DURABLE=false" in path.read_text()
    overlay = (ROOT / "deploy/compose.durable-cleanup.yml").read_text()
    assert 'JOB_CLEANUP_DURABLE: "true"' in overlay
    assert "JOB_ROOT: /data/jobs" in overlay
    assert "- lao-ocr-jobs:/data/jobs" in overlay
    assert "volumes:\n  lao-ocr-jobs:" in overlay


@pytest.mark.parametrize("name", ["Dockerfile.api", "Dockerfile.owned-api"])
def test_api_images_prepare_private_persistent_job_root_for_nonroot_user(name):
    text = (ROOT / name).read_text()
    assert text.count("       /data/jobs") == 3  # mkdir, ownership, permissions
    assert "chown -R 10001:10001" in text
    assert "chmod 0700" in text
    assert "USER 10001:10001" in text
