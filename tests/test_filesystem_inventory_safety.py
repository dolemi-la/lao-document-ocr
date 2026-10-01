"""Metadata failures and races must not become complete or destructive inventories."""

from __future__ import annotations

import builtins
import json
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from test_filesystem_inventory import LEGACY, STAGING, fixture_tree

import services.api.app.filesystem_inventory as module
from services.api.app.filesystem_inventory import InventoryError, InventoryLimits, inventory


def test_only_directory_descriptors_are_opened_and_no_mutators_are_called(tmp_path, monkeypatch):
    root = tmp_path / "root"
    fixture_tree(root)
    original_open = os.open
    calls = []

    def directory_only(path, flags, *args, **kwargs):
        assert flags & os.O_ACCMODE == os.O_RDONLY
        assert flags & os.O_DIRECTORY
        assert flags & os.O_NOFOLLOW
        calls.append(str(path))
        return original_open(path, flags, *args, **kwargs)

    def prohibited(*args, **kwargs):
        raise AssertionError("inventory attempted a content read or mutation")

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", directory_only)
        patch.setattr(builtins, "open", prohibited)
        for name in ("chmod", "fchmod", "mkdir", "unlink", "remove", "rename", "replace", "rmdir"):
            patch.setattr(os, name, prohibited)
        report = inventory(root)
    assert report["traversal_complete"]
    assert len(calls) == report["directories_opened"]
    assert all(not name.endswith((".tmp", ".part", ".zip")) for name in calls)


def test_entry_metadata_failure_is_partial_and_private(tmp_path, monkeypatch):
    root = tmp_path / "PRIVATE-root"
    root.mkdir()
    (root / "PRIVATE-name").touch()
    original = os.stat

    def unavailable(path, *args, **kwargs):
        if path == "PRIVATE-name":
            raise PermissionError("PRIVATE device details")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", unavailable)
    report = inventory(root)
    assert report["status"] == "partial"
    assert report["issues"] == {"entry_unavailable": 1}
    assert report["categories"]["unreadable_entry"]["count"] == 1
    assert report["items"][0]["size_bytes"] is None
    assert "PRIVATE" not in json.dumps(report)


def test_root_enumeration_failure_is_not_an_empty_complete_root(tmp_path, monkeypatch):
    def denied(_):
        raise PermissionError("PRIVATE enumeration error")

    monkeypatch.setattr(os, "scandir", denied)
    report = inventory(tmp_path)
    assert report["status"] == "partial"
    assert report["issues"] == {"directory_unavailable": 1}
    assert report["entries_seen"] == 0
    assert "PRIVATE" not in json.dumps(report)


def test_failed_iterator_keeps_observed_metadata_and_closes(tmp_path, monkeypatch):
    (tmp_path / LEGACY).write_bytes(b"authored")
    original = os.scandir
    closed = []

    class Interrupted:
        def __init__(self, descriptor):
            self.iterator = original(descriptor)
            self.count = 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.iterator.close()
            closed.append(True)

        def __next__(self):
            if self.count:
                raise OSError("PRIVATE iterator error")
            self.count += 1
            return next(self.iterator)

    monkeypatch.setattr(os, "scandir", Interrupted)
    report = inventory(tmp_path)
    assert report["status"] == "partial"
    assert report["entries_seen"] == 1
    assert report["categories"]["legacy_temporary_candidate"]["count"] == 1
    assert closed == [True]


@pytest.mark.parametrize("replacement", ["symlink", "directory"])
def test_directory_replacement_after_stat_is_not_followed(tmp_path, monkeypatch, replacement):
    root = tmp_path / "root"
    root.mkdir()
    child = root / "child"
    child.mkdir()
    (child / "original").touch()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "PRIVATE-foreign").touch()
    original_open = os.open
    changed = []

    def race(path, flags, *args, **kwargs):
        if path == "child" and kwargs.get("dir_fd") is not None and not changed:
            changed.append(True)
            child.rename(tmp_path / "moved")
            if replacement == "symlink":
                child.symlink_to(outside, target_is_directory=True)
            else:
                child.mkdir()
                (child / "PRIVATE-replacement").touch()
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", race)
    report = inventory(root, include_paths=True)
    assert changed == [True]
    assert report["status"] == "partial"
    assert report["directories_opened"] == 1
    assert "PRIVATE" not in json.dumps(report)
    assert sum(report["issues"].values()) == 1


@pytest.mark.parametrize("where", ["root", "child"])
def test_descriptor_is_closed_when_fstat_fails(tmp_path, monkeypatch, where):
    (tmp_path / "child").mkdir()
    original_open, original_close, original_fstat = os.open, os.close, os.fstat
    opened, closed = [], []

    def watched_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        opened.append((descriptor, "child" if path == "child" else "root"))
        return descriptor

    def failed_stat(descriptor):
        if (descriptor, where) in opened:
            raise OSError("PRIVATE metadata failure")
        return original_fstat(descriptor)

    def watched_close(descriptor):
        closed.append(descriptor)
        return original_close(descriptor)

    monkeypatch.setattr(os, "open", watched_open)
    monkeypatch.setattr(os, "fstat", failed_stat)
    monkeypatch.setattr(os, "close", watched_close)
    if where == "root":
        with pytest.raises(InventoryError, match="root_unavailable"):
            inventory(tmp_path)
    else:
        assert inventory(tmp_path)["status"] == "partial"
    assert sorted(fd for fd, _ in opened) == sorted(closed)


def test_cooperative_time_limit_stops_after_slow_metadata_call(tmp_path, monkeypatch):
    for index in range(3):
        (tmp_path / str(index)).touch()
    clock = [0.0]
    original = os.stat
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    def delayed_stat(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] += 2
        return result

    monkeypatch.setattr(os, "stat", delayed_stat)
    report = inventory(tmp_path, limits=InventoryLimits(max_seconds=1))
    assert report["entries_seen"] == 1
    assert report["status"] == "partial"
    assert report["issues"] == {"time_limit": 1}
    assert report["elapsed_seconds"] == 2  # Not a hard I/O interrupt or timeout guarantee.


@pytest.mark.parametrize("reason", ["device_boundary", "directory_revisited"])
def test_device_and_revisited_directory_boundaries_are_reported(tmp_path, monkeypatch, reason):
    child = tmp_path / "child"
    child.mkdir()
    (child / LEGACY).touch()
    original = os.stat
    root_stat = tmp_path.stat()

    def altered(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path == "child":
            fields = {
                name: getattr(result, name)
                for name in (
                    "st_mode",
                    "st_size",
                    "st_nlink",
                    "st_dev",
                    "st_ino",
                    "st_mtime_ns",
                )
            }
            if reason == "device_boundary":
                fields["st_dev"] = root_stat.st_dev + 1
            else:
                fields["st_ino"] = root_stat.st_ino
            return SimpleNamespace(**fields)
        return result

    monkeypatch.setattr(os, "stat", altered)
    report = inventory(tmp_path)
    assert report["issues"] == {reason: 1}
    assert report["directories_opened"] == 1
    assert report["entries_seen"] == 1


def test_overlong_relative_name_is_not_returned_even_in_private_details(tmp_path, monkeypatch):
    (tmp_path / "PRIVATE-long-name").touch()
    monkeypatch.setattr(module, "MAX_RELATIVE_PATH_BYTES", 5)
    report = inventory(tmp_path, include_paths=True)
    assert report["issues"] == {"path_limit": 1}
    assert report["items"][0]["size_bytes"] is None
    assert "relative_path" not in report["items"][0]
    assert "PRIVATE" not in json.dumps(report)


def test_unsupported_platform_fails_before_access(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "_SUPPORTED", False)
    missing = tmp_path / "PRIVATE-missing"
    with pytest.raises(InventoryError, match="unsupported_platform"):
        inventory(missing)
    assert not missing.exists()


def test_active_writer_lock_is_not_acquired_or_released_by_inventory(tmp_path):
    from services.api.app.filesystem_writes import FilesystemWriteBusyError
    from services.api.app.storage import FilesystemArtifactStorage, StoredArtifact

    storage = FilesystemArtifactStorage(tmp_path / "root", recoverable_writes=True)
    artifact = StoredArtifact("jobs/fixture/result.zip", "result.zip", "application/zip", 0)
    destination = storage.root / artifact.key
    helper = storage._recoverable
    with helper._locked(destination, wait=True):
        stage = helper.stage_path(destination)
        stage.write_bytes(b"authored active copy")
        inode = helper.lock_path(destination).stat().st_ino
        with ThreadPoolExecutor(max_workers=1) as pool:
            report = pool.submit(inventory, storage.root).result(timeout=1)
        assert report["traversal_complete"]
        assert report["categories"]["recoverable_stage_candidate"]["count"] == 1
        assert report["ownership_verified"] is False
        with pytest.raises(FilesystemWriteBusyError):
            storage.delete(artifact)
        assert stage.read_bytes() == b"authored active copy"
        assert helper.lock_path(destination).stat().st_ino == inode
    assert (storage.root / STAGING).exists()


def test_non_utf8_filename_is_serializable_without_content_reads(tmp_path, monkeypatch):
    from contextlib import contextmanager

    # Some local filesystems reject non-UTF-8 names. Supply a directory-entry
    # double while using real metadata so serialization is exercised everywhere.
    fixture = tmp_path / "fixture"
    fixture.touch()
    metadata = fixture.stat()
    name = os.fsdecode(b"PRIVATE-\xff.tmp")
    original_stat = os.stat

    @contextmanager
    def entries(_):
        yield iter([SimpleNamespace(name=name)])

    def entry_stat(path, *args, **kwargs):
        if path == name:
            return metadata
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", entries)
    monkeypatch.setattr(os, "stat", entry_stat)
    report = inventory(tmp_path)
    assert report["traversal_complete"]
    assert "PRIVATE" not in json.dumps(report)
    private = inventory(tmp_path, include_paths=True)
    assert os.fsencode(private["items"][0]["relative_path"]) == b"PRIVATE-\xff.tmp"
    json.dumps(private, ensure_ascii=True)
