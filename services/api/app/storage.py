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


def _normalize_s3_prefix(prefix: str) -> str:
    value = prefix.strip().strip("/")
    if not value:
        return ""
    path = _safe_relative_key(value)
    return "/".join(path.parts)


def _s3_not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error")
    metadata = response.get("ResponseMetadata")
    code = str(error.get("Code")) if isinstance(error, dict) else ""
    status = (
        metadata.get("HTTPStatusCode")
        if isinstance(metadata, dict)
        else None
    )
    return code in {"404", "NoSuchKey", "NotFound"} or status == 404


class S3ArtifactStorage:
    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        endpoint_url: str | None = None,
        region_name: str | None = None,
        force_path_style: bool = False,
        client=None,
    ) -> None:
        if not bucket.strip():
            raise ValueError("S3 bucket must not be empty")
        self.bucket = bucket.strip()
        self.prefix = _normalize_s3_prefix(prefix)
        self.endpoint_url = endpoint_url or None
        self.region_name = region_name or None
        self.force_path_style = bool(force_path_style)

        if client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:
                raise RuntimeError(
                    "S3 result storage requires optional dependencies. "
                    "Install with: pip install -e '.[s3]'"
                ) from exc

            config = Config(
                s3={
                    "addressing_style": (
                        "path" if self.force_path_style else "auto"
                    )
                }
            )
            client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                region_name=self.region_name,
                config=config,
            )
        self.client = client

    def _object_key(self, key: str) -> str:
        relative = _safe_relative_key(key)
        normalized = "/".join(relative.parts)
        if not self.prefix:
            return normalized
        return f"{self.prefix}/{normalized}"

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

        object_key = self._object_key(key)
        safe_filename = Path(filename).name or "artifact.bin"
        self.client.upload_file(
            str(source_path),
            self.bucket,
            object_key,
            ExtraArgs={"ContentType": media_type},
        )
        return StoredArtifact(
            key=key,
            filename=safe_filename,
            media_type=media_type,
            size_bytes=source_path.stat().st_size,
        )

    def exists(self, artifact: StoredArtifact) -> bool:
        try:
            self.client.head_object(
                Bucket=self.bucket,
                Key=self._object_key(artifact.key),
            )
            return True
        except Exception as exc:
            if _s3_not_found(exc):
                return False
            raise

    def iter_bytes(
        self,
        artifact: StoredArtifact,
        *,
        chunk_size: int = 1024 * 1024,
    ) -> Iterator[bytes]:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        response = self.client.get_object(
            Bucket=self.bucket,
            Key=self._object_key(artifact.key),
        )
        body = response["Body"]
        try:
            if hasattr(body, "iter_chunks"):
                for chunk in body.iter_chunks(chunk_size=chunk_size):
                    if chunk:
                        yield chunk
            else:
                while chunk := body.read(chunk_size):
                    yield chunk
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()

    def delete(self, artifact: StoredArtifact) -> None:
        self.client.delete_object(
            Bucket=self.bucket,
            Key=self._object_key(artifact.key),
        )

    def metadata(self) -> dict:
        return {
            "backend": "s3",
            "bucket": self.bucket,
            "prefix": self.prefix,
            "endpoint_url": self.endpoint_url,
            "region_name": self.region_name,
            "force_path_style": self.force_path_style,
        }
