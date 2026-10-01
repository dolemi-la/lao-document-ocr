"""CLI and SDK configuration tests, without credentials or provider traffic."""

from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
from test_s3_multipart_inventory import BUCKET, PREFIX, ListingClient, page, upload

import services.api.app.s3_multipart_inventory as module

ROOT = Path(__file__).parents[1]
MODULE = "services.api.app.s3_multipart_inventory"
ARGS = ["--bucket", BUCKET, "--prefix", PREFIX]


class ClosableClient(ListingClient):
    def __init__(self, pages):
        super().__init__(pages)
        self.closed = False

    def close(self):
        self.closed = True


def test_sdk_client_is_lazy_and_uses_explicit_bounded_transport_configuration(monkeypatch):
    created = []
    boto = types.ModuleType("boto3")
    config = types.ModuleType("botocore.config")
    core = types.ModuleType("botocore")
    config.Config = lambda **kwargs: kwargs
    boto.client = lambda *args, **kwargs: created.append((args, kwargs)) or "client"
    monkeypatch.setitem(sys.modules, "boto3", boto)
    monkeypatch.setitem(sys.modules, "botocore", core)
    monkeypatch.setitem(sys.modules, "botocore.config", config)
    result = module.build_client(
        endpoint_url="https://fixture.invalid",
        region="auto",
        force_path_style=True,
    )
    assert result == "client"
    assert created == [
        (
            ("s3",),
            {
                "endpoint_url": "https://fixture.invalid",
                "region_name": "auto",
                "config": {
                    "connect_timeout": 3,
                    "read_timeout": 5,
                    "max_pool_connections": 1,
                    "retries": {"mode": "standard", "total_max_attempts": 1},
                    "ignore_configured_endpoint_urls": True,
                    "s3": {"addressing_style": "path"},
                },
            },
        )
    ]


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://fixture.invalid",
        "https://PRIVATE:secret@fixture.invalid",
        "https://",
        "https://fixture.invalid?PRIVATE=secret",
        "https://fixture.invalid/#PRIVATE",
        "https://fixture.invalid/path",
        "https://fixture.invalid:0",
        "https://fixture.invalid:99999",
        "https://fixture.invalid\n",
        "https://fixture.invalid:PRIVATE",
        "PRIVATE" * 400,
    ],
)
def test_invalid_connection_is_rejected_before_importing_sdk(monkeypatch, endpoint):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name not in {"boto3", "botocore.config"}, "Invalid input reached SDK import"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(module.InventoryError, match="invalid_arguments"):
        module.build_client(endpoint_url=endpoint)


def test_missing_optional_sdk_has_a_fixed_error(monkeypatch):
    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError("PRIVATE local dependency path")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(module.InventoryError, match="s3_dependencies_unavailable") as error:
        module.build_client()
    assert "PRIVATE" not in str(error.value)


def test_sdk_initialization_error_is_private(monkeypatch):
    boto, config, core = (
        types.ModuleType(name) for name in ("boto3", "botocore.config", "botocore")
    )
    config.Config = lambda **kwargs: kwargs

    def fail(*args, **kwargs):
        raise RuntimeError("PRIVATE credential and endpoint configuration")

    boto.client = fail
    monkeypatch.setitem(sys.modules, "boto3", boto)
    monkeypatch.setitem(sys.modules, "botocore", core)
    monkeypatch.setitem(sys.modules, "botocore.config", config)
    with pytest.raises(module.InventoryError, match="client_unavailable") as error:
        module.build_client()
    assert "PRIVATE" not in str(error.value)


@pytest.mark.parametrize(
    "responses,exit_code",
    [
        ([page([upload()])], 0),
        ([OSError("PRIVATE provider body")], 3),
        ([{}], 3),
    ],
)
def test_cli_closes_its_client_and_preserves_partial_exit_code(
    monkeypatch, capsys, responses, exit_code
):
    client = ClosableClient(responses)
    monkeypatch.setattr(module, "build_client", lambda **kwargs: client)
    assert module.main(ARGS) == exit_code
    output = capsys.readouterr()
    assert output.err == ""
    assert "PRIVATE" not in output.out
    assert client.closed
    assert json.loads(output.out)["status"] == ("complete" if exit_code == 0 else "partial")


def test_cli_explicit_key_disclosure_still_excludes_upload_ids(monkeypatch, capsys):
    key = PREFIX + "PRIVATE-ລາວ\n%2F+.zip"
    client = ClosableClient([page([upload(key, "DO-NOT-SHOW-UPLOAD-ID")])])
    monkeypatch.setattr(module, "build_client", lambda **kwargs: client)
    assert module.main([*ARGS, "--include-keys"]) == 0
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report["items"][0]["key"] == key
    assert "DO-NOT-SHOW-UPLOAD-ID" not in output.out
    assert "\\n" in output.out
    assert output.err == ""


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--bucket", "PRIVATE"],
        [*ARGS, "--max-uploads", "PRIVATE"],
        [*ARGS, "--max-uploads", "0"],
        [*ARGS, "--max-seconds", "nan"],
        [*ARGS, "--abort"],
        [*ARGS, "--delete"],
        [*ARGS, "--include-key"],
        [*ARGS, "--output", "PRIVATE-file"],
        [*ARGS, "--expected-bucket-owner", "PRIVATE"],
    ],
)
def test_invalid_cli_cannot_build_client_or_echo_argv(monkeypatch, capsys, args):
    def forbidden(**kwargs):
        raise AssertionError("Invalid args must not reach credentials or network setup")

    monkeypatch.setattr(module, "build_client", forbidden)
    assert module.main(args) == 2
    output = capsys.readouterr()
    assert output.err == ""
    assert "PRIVATE" not in output.out
    assert json.loads(output.out)["status"] == "error"


def test_cli_close_failure_does_not_leak_sdk_error(monkeypatch, capsys):
    client = ClosableClient([page()])

    def close():
        client.closed = True
        raise OSError("PRIVATE connection body")

    monkeypatch.setattr(client, "close", close)
    monkeypatch.setattr(module, "build_client", lambda **kwargs: client)
    assert module.main(ARGS) == 0
    assert client.closed
    output = capsys.readouterr()
    assert "PRIVATE" not in output.out + output.err


def test_clean_import_and_help_do_not_start_api_or_load_credentials(tmp_path):
    env = {
        **os.environ,
        "JOB_ROOT": str(tmp_path / "jobs"),
        "RESULT_STORAGE_ROOT": str(tmp_path / "results"),
        "JOB_CLEANUP_DURABLE": "invalid",
    }
    code = f"""
import sys
import {MODULE} as module
assert 'boto3' not in sys.modules
assert 'botocore' not in sys.modules
assert 'services.api.app.main' not in sys.modules
assert 'services.api.app.storage' not in sys.modules
assert 'services.api.app.jobs' not in sys.modules
module.main(['--help'])
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "--include-keys" in result.stdout
    assert "--prefix" in result.stdout
    assert not list(tmp_path.iterdir())
    assert result.stderr == ""
