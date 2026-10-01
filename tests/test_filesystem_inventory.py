"""Authored metadata fixtures; inventory must not decide which files to delete."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import replace

import pytest

from services.api.app.filesystem_inventory import InventoryError, InventoryLimits, inventory

STAGING = ".lao-ocr-writes-v1"
LEGACY = ".private-source-ocr.zip." + "a" * 32 + ".tmp"


def fixture_tree(root):
    root.mkdir(mode=0o750)
    folder = root / "jobs" / "fixture"
    folder.mkdir(parents=True)
    (folder / LEGACY).write_bytes(b"authored partial copy")
    (folder / "result.zip").write_bytes(b"authored result")
    (folder / ".result.zip.bad.tmp").write_bytes(b"not the generated pattern")
    staging = root / STAGING
    staging.mkdir()
    (staging / ("b" * 64 + ".part")).write_bytes(b"authored stage")
    (staging / "lock-3f").touch()
    (staging / "lock-40").write_bytes(b"not a valid slot")
    (staging / "notes.part").write_bytes(b"not a key digest")
    return folder, staging


def test_classification_is_metadata_only_and_paths_are_private_by_default(tmp_path):
    root = tmp_path / "PRIVATE-root"
    folder, _ = fixture_tree(root)
    (folder / "link").symlink_to(tmp_path / "PRIVATE-target")
    os.mkfifo(folder / "pipe")
    report = inventory(root)
    assert report["status"] == "complete"
    assert report["traversal_complete"]
    assert report["snapshot"] is False
    assert report["ownership_verified"] is False
    assert report["deletion_authorized"] is False
    counts = report["categories"]
    assert counts["legacy_temporary_candidate"]["count"] == 1
    assert counts["recoverable_stage_candidate"]["count"] == 1
    assert counts["coordination_lock_candidate"]["count"] == 1
    assert counts["symlink"]["count"] == counts["special_file"]["count"] == 1
    assert counts["directory"]["count"] == 3
    assert counts["other_regular_file"]["count"] == 4
    encoded = json.dumps(report)
    assert "PRIVATE" not in encoded
    assert "private-source" not in encoded
    assert "authored" not in encoded
    assert str(root) not in encoded
    assert all("relative_path" not in item for item in report["items"])


def test_explicit_paths_are_relative_and_json_escaped(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    name = 'private-ລາວ\n".txt'
    (root / name).write_bytes(b"authored")
    report = inventory(root, include_paths=True)
    item = report["items"][0]
    assert item["relative_path"] == name
    assert item["path_id"] == hashlib.sha256(os.fsencode(name)).hexdigest()
    assert str(root) not in json.dumps(report)
    assert json.loads(json.dumps(report))["items"][0]["relative_path"] == name


def test_inventory_never_changes_contents_permissions_or_directory_names(tmp_path):
    root = tmp_path / "root"
    fixture_tree(root)

    def snapshot():
        result = {}
        for path in [root, *root.rglob("*")]:
            info = path.lstat()
            result[str(path.relative_to(root))] = (
                info.st_mode,
                info.st_ino,
                info.st_mtime_ns,
                path.read_bytes() if path.is_file() else None,
            )
        return result

    before = snapshot()
    assert inventory(root)["traversal_complete"]
    assert snapshot() == before
    assert stat.S_IMODE(root.stat().st_mode) == 0o750


def test_symlink_directory_is_observed_but_never_followed(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / LEGACY).write_bytes(b"do not inspect")
    (root / "elsewhere").symlink_to(outside, target_is_directory=True)
    report = inventory(root, include_paths=True)
    assert report["traversal_complete"]
    assert report["entries_seen"] == 1
    assert report["directories_opened"] == 1
    assert report["categories"]["symlink"]["count"] == 1
    assert report["categories"]["legacy_temporary_candidate"]["count"] == 0
    assert report["items"][0]["relative_path"] == "elsewhere"


def test_hardlinks_are_not_treated_as_independent_temporary_candidates(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    source = root / "original"
    source.write_bytes(b"authored shared inode")
    os.link(source, root / LEGACY)
    report = inventory(root)
    assert report["categories"]["hardlinked_file"]["count"] == 2
    assert report["categories"]["legacy_temporary_candidate"]["count"] == 0
    assert all(item["link_count"] == 2 for item in report["items"])


@pytest.mark.parametrize(
    "name",
    [
        ".x." + "A" * 32 + ".tmp",
        ".x." + "a" * 31 + ".tmp",
        "." + "a" * 32 + ".tmp",
        "x." + "a" * 32 + ".tmp",
        ".x." + "a" * 32 + ".tmp.extra",
        "b" * 64 + ".part",
        "lock-00",
    ],
)
def test_near_matches_and_staging_names_outside_namespace_remain_other_files(tmp_path, name):
    root = tmp_path / "root"
    root.mkdir()
    (root / name).write_bytes(b"authored")
    report = inventory(root)
    assert report["categories"]["other_regular_file"]["count"] == 1


def test_entry_limit_is_reported_without_claiming_complete_totals(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    for index in range(5):
        (root / str(index)).write_bytes(b"x")
    report = inventory(root, limits=InventoryLimits(max_entries=2))
    assert report["status"] == "partial"
    assert not report["traversal_complete"]
    assert report["entries_seen"] == 2
    assert report["issues"]["entry_limit"] == 1
    assert report["categories"]["other_regular_file"]["count"] == 2


def test_item_limit_only_truncates_details_not_summary_traversal(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    for index in range(5):
        (root / str(index)).write_bytes(b"x")
    report = inventory(root, limits=InventoryLimits(max_items=1))
    assert report["traversal_complete"]
    assert report["entries_seen"] == 5
    assert len(report["items"]) == 1
    assert report["items_omitted"] == 4
    assert report["details_complete"] is False
    assert report["categories"]["other_regular_file"]["count"] == 5


def test_zero_details_is_supported_without_skipping_metadata(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / LEGACY).write_bytes(b"x")
    report = inventory(root, limits=InventoryLimits(max_items=0))
    assert report["items"] == []
    assert report["items_omitted"] == 1
    assert report["traversal_complete"]
    assert report["categories"]["legacy_temporary_candidate"]["count"] == 1


def test_depth_limit_marks_unvisited_subtrees_even_when_their_size_is_unknown(tmp_path):
    root = tmp_path / "root"
    (root / "child" / "nested").mkdir(parents=True)
    (root / "child" / "nested" / LEGACY).write_bytes(b"x")
    report = inventory(root, limits=InventoryLimits(max_depth=1))
    assert report["status"] == "partial"
    assert report["entries_seen"] == 1
    assert report["issues"]["depth_limit"] == 1
    assert report["categories"]["legacy_temporary_candidate"]["count"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_entries", 0),
        ("max_entries", True),
        ("max_entries", 100001),
        ("max_items", -1),
        ("max_items", 2001),
        ("max_items", 1.5),
        ("max_depth", 0),
        ("max_depth", 65),
        ("max_depth", True),
        ("max_seconds", 0),
        ("max_seconds", 61),
        ("max_seconds", float("nan")),
        ("max_seconds", float("inf")),
        ("max_seconds", True),
        ("max_seconds", "5"),
    ],
)
def test_bad_limits_fail_before_touching_root(tmp_path, field, value):
    missing = tmp_path / "PRIVATE-missing"
    with pytest.raises(InventoryError) as error:
        inventory(missing, limits=replace(InventoryLimits(), **{field: value}))
    assert error.value.code == "invalid_limits"
    assert "PRIVATE" not in str(error.value)
    assert not missing.exists()


@pytest.mark.parametrize("kind", ["missing", "file", "symlink", "symlink-slash"])
def test_invalid_root_is_private_and_never_created_or_followed(tmp_path, kind):
    root = tmp_path / "PRIVATE-root"
    if kind == "file":
        root.write_bytes(b"authored")
    elif kind.startswith("symlink"):
        target = tmp_path / "target"
        target.mkdir()
        root.symlink_to(target, target_is_directory=True)
    supplied = str(root) + ("/" if kind == "symlink-slash" else "")
    with pytest.raises(InventoryError) as error:
        inventory(supplied)
    assert error.value.code == "root_unavailable"
    assert "PRIVATE" not in str(error.value)
    if kind == "missing":
        assert not root.exists()


def test_empty_root_is_complete_not_an_error(tmp_path):
    report = inventory(tmp_path)
    assert report["status"] == "complete"
    assert report["entries_seen"] == 0
    assert report["items"] == []
    assert report["issues"] == {}
    assert all(row == {"count": 0, "logical_bytes": 0} for row in report["categories"].values())
