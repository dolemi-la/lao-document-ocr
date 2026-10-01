"""The standalone operator command must not start the API or create storage."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_filesystem_inventory import LEGACY

MODULE = "services.api.app.filesystem_inventory"
ROOT = Path(__file__).parents[1]


def invoke(arguments, *, env=None):
    return subprocess.run(
        [sys.executable, "-B", "-m", MODULE, *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    )


def test_cli_does_not_import_api_or_create_configured_service_roots(tmp_path):
    root = tmp_path / "PRIVATE-result-root"
    root.mkdir()
    (root / LEGACY).write_bytes(b"authored fixture")
    jobs = tmp_path / "PRIVATE-jobs"
    results = tmp_path / "PRIVATE-results"
    env = {
        **os.environ,
        "JOB_ROOT": str(jobs),
        "RESULT_STORAGE_ROOT": str(results),
        "JOB_CLEANUP_DURABLE": "invalid-if-the-api-were-imported",
        "RESULT_STORAGE_BACKEND": "invalid-if-instantiated",
    }
    result = invoke(["--root", str(root)], env=env)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["traversal_complete"]
    assert report["categories"]["legacy_temporary_candidate"]["count"] == 1
    assert "PRIVATE" not in result.stdout + result.stderr
    assert "private-source" not in result.stdout
    assert result.stderr == ""
    assert not jobs.exists()
    assert not results.exists()
    assert list(root.iterdir()) == [root / LEGACY]


def test_cli_has_distinct_exit_code_for_partial_scan(tmp_path):
    (tmp_path / "a").touch()
    (tmp_path / "b").touch()
    result = invoke(["--root", str(tmp_path), "--max-entries", "1"])
    assert result.returncode == 3
    report = json.loads(result.stdout)
    assert report["status"] == "partial"
    assert report["entries_seen"] == 1
    assert report["issues"] == {"entry_limit": 1}
    assert result.stderr == ""


def test_cli_reports_detail_truncation_independently(tmp_path):
    (tmp_path / "a").touch()
    (tmp_path / "b").touch()
    result = invoke(["--root", str(tmp_path), "--max-items", "0"])
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["traversal_complete"]
    assert not report["details_complete"]
    assert report["items_omitted"] == 2


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--root", "PRIVATE-missing"],
        ["--root", "PRIVATE-missing", "--max-entries", "PRIVATE-invalid"],
        ["--root", "PRIVATE-missing", "--max-seconds", "nan"],
        ["--root", "PRIVATE-missing", "--delete"],
        ["--root", "PRIVATE-missing", "--output", "PRIVATE-output.json"],
        ["--root", "PRIVATE-missing", "--include-path"],
    ],
)
def test_errors_are_machine_readable_and_do_not_echo_paths(arguments):
    result = invoke(arguments)
    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["status"] == "error"
    assert report["error"] in {"root_unavailable", "invalid_arguments", "invalid_limits"}
    assert "PRIVATE" not in result.stdout + result.stderr
    assert result.stderr == ""


def test_help_is_available_without_selecting_or_scanning_a_root():
    result = invoke(["--help"])
    assert result.returncode == 0
    assert "--include-paths" in result.stdout
    assert "--max-entries" in result.stdout
    assert "--max-seconds" in result.stdout
    assert result.stderr == ""


def test_cli_path_disclosure_requires_explicit_flag(tmp_path):
    name = 'PRIVATE-ລາວ\n".txt'
    (tmp_path / name).write_bytes(b"authored")
    result = invoke(["--root", str(tmp_path), "--include-paths"])
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["paths_included"] is True
    assert report["items"][0]["relative_path"] == name
    assert str(tmp_path) not in result.stdout
    assert "authored" not in result.stdout
