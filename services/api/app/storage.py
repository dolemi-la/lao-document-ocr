from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class StoredArtifact:
    key: str
    filename: str
    media_type: str
    size_bytes: int


class ArtifactStorage(Protocol):
    def put_file(
        self,
        source: str | Path,
        *,
        key: str,
        filename: str,
        media_type: str,
    ) -> StoredArtifact: ...

    def exists(self, artifact: StoredArtifact) -> bool: ...

    def iter_bytes(
        self,
        artifact: StoredArtifact,
        *,
        chunk_size: int = 1024 * 1024,
    ) -> Iterator[bytes]: ...

    def delete(self, artifact: StoredArtifact) -> None: ...

    def metadata(self) -> dict: ...


def _safe_relative_key(key: str) -> Path:
    path = Path(key)
    if not key or path.is_absolute() or ".." in path.parts:
        raise ValueError("storage key must be a safe relative path")
    if any(part in {"", "."} for part in path.parts):
        raise ValueError("storage key contains an invalid path segment")
    return path


class FilesystemArtifactStorage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)

    def _path(self, key: str) -> Path:
        relative = _safe_relative_key(key)
        destination = (self.root / relative).resolve()
        try:
            destination.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("storage key escapes storage root") from exc
        return destination

    def put_file(
        self,
        source: str | Path,
        *,
        key: str,
        filename: str,
        media_type: str,
    ) -> StoredArtifact:
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"artifact source not found: {source_path}")

        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination.parent.chmod(0o700)
        temporary = destination.with_name(
            f".{destination.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            shutil.copy2(source_path, temporary)
            temporary.chmod(0o600)
            temporary.replace(destination)
            destination.chmod(0o600)
        finally:
            temporary.unlink(missing_ok=True)
        return StoredArtifact(
            key=key,
            filename=Path(filename).name or "artifact.bin",
            media_type=media_type,
            size_bytes=destination.stat().st_size,
        )

    def exists(self, artifact: StoredArtifact) -> bool:
        return self._path(artifact.key).is_file()

    def iter_bytes(
        self,
        artifact: StoredArtifact,
        *,
        chunk_size: int = 1024 * 1024,
    ) -> Iterator[bytes]:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        path = self._path(artifact.key)
        with path.open("rb") as source:
            while chunk := source.read(chunk_size):
                yield chunk

    def delete(self, artifact: StoredArtifact) -> None:
        path = self._path(artifact.key)
        path.unlink(missing_ok=True)

        parent = path.parent
        while parent != self.root:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    def metadata(self) -> dict:
        return {
            "backend": "filesystem",
            "root": str(self.root),
        }
