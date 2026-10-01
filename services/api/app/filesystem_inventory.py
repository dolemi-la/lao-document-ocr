"""Bounded, metadata-only inventory of an explicitly selected result directory.

Pattern matches are candidates, not ownership, abandonment, or deletion evidence.
Do not construct storage/managers here: their constructors can create/chmod paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from services.api.app.filesystem_writes import STAGING_DIRECTORY

SCHEMA = "filesystem-remnant-inventory/v1"
MAX_RELATIVE_PATH_BYTES = 8192
CATEGORIES = (
    "directory",
    "legacy_temporary_candidate",
    "recoverable_stage_candidate",
    "coordination_lock_candidate",
    "other_regular_file",
    "hardlinked_file",
    "symlink",
    "special_file",
    "unreadable_entry",
)
_LEGACY = re.compile(r"\..+\.[0-9a-f]{32}\.tmp\Z", re.DOTALL)
_STAGE = re.compile(r"[0-9a-f]{64}\.part\Z")
_SLOT = re.compile(r"lock-[0-3][0-9a-f]\Z")
_SUPPORTED = (
    os.name == "posix"
    and all(hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC"))
    and os.scandir in os.supports_fd
    and os.open in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.stat in os.supports_follow_symlinks
)


class InventoryError(RuntimeError):
    """Fixed codes only; never include operator paths or operating-system errors."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class InventoryLimits:
    max_entries: int = 10000
    max_items: int = 200
    max_depth: int = 8
    max_seconds: float = 5.0

    def validate(self) -> None:
        for value, low, high in (
            (self.max_entries, 1, 100000),
            (self.max_items, 0, 2000),
            (self.max_depth, 1, 64),
        ):
            if type(value) is not int or not low <= value <= high:
                raise InventoryError("invalid_limits")
        if type(self.max_seconds) not in (int, float) or not 0 < self.max_seconds <= 60:
            raise InventoryError("invalid_limits")


def _category(relative: str, info: os.stat_result) -> str:
    if stat.S_ISDIR(info.st_mode):
        return "directory"
    if stat.S_ISLNK(info.st_mode):
        return "symlink"
    if not stat.S_ISREG(info.st_mode):
        return "special_file"
    if info.st_nlink != 1:
        return "hardlinked_file"
    parts = relative.split("/")
    if len(parts) == 2 and parts[0] == STAGING_DIRECTORY:
        if _STAGE.fullmatch(parts[1]):
            return "recoverable_stage_candidate"
        if _SLOT.fullmatch(parts[1]):
            return "coordination_lock_candidate"
    if _LEGACY.fullmatch(parts[-1]):
        return "legacy_temporary_candidate"
    return "other_regular_file"


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK


class _Inventory:
    def __init__(self, limits: InventoryLimits, include_paths: bool) -> None:
        self.limits = limits
        self.include_paths = include_paths
        self.started = time.monotonic()
        self.entries_seen = 0
        self.directories_opened = 1
        self.categories = {name: {"count": 0, "logical_bytes": 0} for name in CATEGORIES}
        self.items: list[dict] = []
        self.issues: Counter[str] = Counter()
        self.seen_directories: set[tuple[int, int]] = set()
        self.root_device: int | None = None
        self.stopped = False

    def stop_at_budget(self) -> bool:
        if self.stopped:
            return True
        if time.monotonic() - self.started >= self.limits.max_seconds:
            self.issues["time_limit"] += 1
            self.stopped = True
        elif self.entries_seen >= self.limits.max_entries:
            # Conservative at the exact cap: do not enumerate an extra entry
            # merely to prove EOF. A capped report must not claim completeness.
            self.issues["entry_limit"] += 1
            self.stopped = True
        return self.stopped

    def observe(self, relative: str, info: os.stat_result | None) -> None:
        category = _category(relative, info) if info is not None else "unreadable_entry"
        size = info.st_size if info is not None and stat.S_ISREG(info.st_mode) else None
        self.categories[category]["count"] += 1
        self.categories[category]["logical_bytes"] += max(0, size or 0)
        if len(self.items) >= self.limits.max_items:
            return
        item = {
            "path_id": hashlib.sha256(os.fsencode(relative)).hexdigest(),
            "category": category,
            "size_bytes": size,
            "mtime_ns": info.st_mtime_ns if info is not None else None,
            "link_count": info.st_nlink if info is not None else None,
        }
        if self.include_paths and len(os.fsencode(relative)) <= MAX_RELATIVE_PATH_BYTES:
            item["relative_path"] = relative
        self.items.append(item)

    def walk(self, descriptor: int, prefix: str, depth: int) -> None:
        if self.stop_at_budget():
            return
        try:
            with os.scandir(descriptor) as entries:
                while not self.stop_at_budget():
                    try:
                        entry = next(entries)
                    except StopIteration:
                        break
                    self.entries_seen += 1
                    relative = f"{prefix}/{entry.name}" if prefix else entry.name
                    if len(os.fsencode(relative)) > MAX_RELATIVE_PATH_BYTES:
                        self.issues["path_limit"] += 1
                        self.observe(relative, None)
                        continue
                    try:
                        info = os.stat(entry.name, dir_fd=descriptor, follow_symlinks=False)
                    except OSError:
                        self.issues["entry_unavailable"] += 1
                        self.observe(relative, None)
                        continue
                    self.observe(relative, info)
                    if not stat.S_ISDIR(info.st_mode):
                        continue
                    if info.st_dev != self.root_device:
                        self.issues["device_boundary"] += 1
                        continue
                    if depth >= self.limits.max_depth:
                        self.issues["depth_limit"] += 1
                        continue
                    if _identity(info) in self.seen_directories:
                        self.issues["directory_revisited"] += 1
                        continue
                    if self.stop_at_budget():
                        break
                    self.walk_child(descriptor, entry.name, relative, depth, info)
        except OSError:
            # Iteration/opening failures are incomplete evidence, not empty trees.
            self.issues["directory_unavailable"] += 1

    def walk_child(
        self,
        parent: int,
        name: str,
        relative: str,
        depth: int,
        before: os.stat_result,
    ) -> None:
        child = None
        try:
            child = os.open(name, _directory_flags(), dir_fd=parent)
            after = os.fstat(child)
            if not stat.S_ISDIR(after.st_mode) or _identity(after) != _identity(before):
                self.issues["directory_changed"] += 1
                return
            self.seen_directories.add(_identity(after))
            self.directories_opened += 1
            self.walk(child, relative, depth + 1)
        except OSError:
            self.issues["directory_unavailable"] += 1
        finally:
            if child is not None:
                os.close(child)

    def report(self) -> dict:
        omitted = self.entries_seen - len(self.items)
        return {
            "schema": SCHEMA,
            "mode": "read-only-metadata",
            "status": "partial" if self.issues else "complete",
            "traversal_complete": not self.issues,
            "snapshot": False,
            "ownership_verified": False,
            "deletion_authorized": False,
            "limits": asdict(self.limits),
            "elapsed_seconds": round(max(0.0, time.monotonic() - self.started), 6),
            "entries_seen": self.entries_seen,
            "directories_opened": self.directories_opened,
            "categories": self.categories,
            "issues": dict(sorted(self.issues.items())),
            "details_complete": omitted == 0,
            "items_omitted": omitted,
            "paths_included": self.include_paths,
            "selection_order": "filesystem-enumeration",
            "items": sorted(self.items, key=lambda item: item["path_id"]),
        }


def inventory(
    root: str | Path,
    *,
    limits: InventoryLimits | None = None,
    include_paths: bool = False,
) -> dict:
    """Inspect names/stat metadata only; never open file contents or acquire locks.

    The root itself must not be a symlink; its operator-selected ancestors use
    ordinary OS resolution. Below that anchored directory, links are not followed.
    Elapsed limits are cooperative and cannot interrupt a stalled filesystem call.
    """
    limits = InventoryLimits() if limits is None else limits
    if not isinstance(limits, InventoryLimits):
        raise InventoryError("invalid_limits")
    limits.validate()
    if type(include_paths) is not bool:
        raise InventoryError("invalid_arguments")
    if not _SUPPORTED:
        raise InventoryError("unsupported_platform")
    if not isinstance(root, (str, Path)) or not str(root):
        raise InventoryError("root_unavailable")
    scan = _Inventory(limits, include_paths)
    descriptor = None
    try:
        # Path strips trailing slashes so O_NOFOLLOW cannot be bypassed by one.
        descriptor = os.open(Path(root), _directory_flags())
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise InventoryError("root_unavailable")
        scan.root_device = info.st_dev
        scan.seen_directories.add(_identity(info))
        scan.walk(descriptor, "", 1)
    except (OSError, ValueError, UnicodeError):
        raise InventoryError("root_unavailable") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return scan.report()


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        # argparse's default embeds untrusted argv (including paths) in stderr.
        raise InventoryError("invalid_arguments")


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", required=True, help="Existing result directory; never created.")
    parser.add_argument("--max-entries", type=int, default=10000)
    parser.add_argument("--max-items", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=5.0)
    parser.add_argument(
        "--include-paths", action="store_true", help="Include private relative names."
    )
    try:
        args = parser.parse_args(argv)
        report = inventory(
            args.root,
            limits=InventoryLimits(
                args.max_entries, args.max_items, args.max_depth, args.max_seconds
            ),
            include_paths=args.include_paths,
        )
    except InventoryError as exc:
        print(json.dumps({"schema": SCHEMA, "status": "error", "error": exc.code}, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2))
    return 0 if report["traversal_complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
