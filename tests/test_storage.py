from pathlib import Path

import pytest

from services.api.app.storage import FilesystemArtifactStorage


def test_filesystem_storage_round_trip(tmp_path) -> None:
    source = tmp_path / "source.zip"
    source.write_bytes(b"abcdefgh")
    storage = FilesystemArtifactStorage(tmp_path / "storage")

    artifact = storage.put_file(
        source,
        key="jobs/job-1/result.zip",
        filename="result.zip",
        media_type="application/zip",
    )

    assert artifact.key == "jobs/job-1/result.zip"
    assert artifact.filename == "result.zip"
    assert artifact.size_bytes == 8
    assert storage.exists(artifact) is True
    assert b"".join(storage.iter_bytes(artifact, chunk_size=3)) == b"abcdefgh"

    storage.delete(artifact)
    assert storage.exists(artifact) is False


def test_storage_rejects_unsafe_keys(tmp_path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"x")
    storage = FilesystemArtifactStorage(tmp_path / "storage")

    for key in ("../escape.bin", "/absolute.bin", "a/../../escape.bin"):
        with pytest.raises(ValueError, match="storage key"):
            storage.put_file(
                source,
                key=key,
                filename="x.bin",
                media_type="application/octet-stream",
            )


def test_storage_requires_existing_source(tmp_path) -> None:
    storage = FilesystemArtifactStorage(tmp_path / "storage")
    with pytest.raises(FileNotFoundError, match="source"):
        storage.put_file(
            tmp_path / "missing.zip",
            key="jobs/a.zip",
            filename="a.zip",
            media_type="application/zip",
        )


def test_storage_metadata(tmp_path) -> None:
    storage = FilesystemArtifactStorage(tmp_path / "storage")
    metadata = storage.metadata()
    assert metadata["backend"] == "filesystem"
    assert metadata["root"].endswith("storage")


def test_storage_artifact_uses_private_permissions(tmp_path) -> None:
    source = tmp_path / "source.zip"
    source.write_bytes(b"private")
    storage = FilesystemArtifactStorage(tmp_path / "storage-private")
    artifact = storage.put_file(
        source,
        key="jobs/job-1/result.zip",
        filename="result.zip",
        media_type="application/zip",
    )
    destination = storage.root / artifact.key
    assert storage.root.stat().st_mode & 0o777 == 0o700
    assert destination.stat().st_mode & 0o777 == 0o600


def test_storage_tightens_existing_root_permissions(tmp_path) -> None:
    root = tmp_path / "existing-storage"
    root.mkdir(mode=0o755)
    root.chmod(0o755)

    storage = FilesystemArtifactStorage(root)

    assert storage.root.stat().st_mode & 0o777 == 0o700


class FakeS3Body:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.closed = False

    def iter_chunks(self, chunk_size: int):
        for index in range(0, len(self.data), chunk_size):
            yield self.data[index : index + chunk_size]

    def close(self) -> None:
        self.closed = True


class FakeS3Error(Exception):
    def __init__(self, code: str, status: int) -> None:
        super().__init__(code)
        self.response = {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict] = {}
        self.last_body: FakeS3Body | None = None

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        data = Path(filename).read_bytes()
        self.objects[(bucket, key)] = {
            "data": data,
            "extra": ExtraArgs or {},
        }

    def head_object(self, *, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise FakeS3Error("NoSuchKey", 404)
        return {"ContentLength": len(self.objects[(Bucket, Key)]["data"])}

    def get_object(self, *, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise FakeS3Error("NoSuchKey", 404)
        self.last_body = FakeS3Body(self.objects[(Bucket, Key)]["data"])
        return {"Body": self.last_body}

    def delete_object(self, *, Bucket, Key):
        self.objects.pop((Bucket, Key), None)
        return {}


def test_s3_storage_round_trip_with_prefix(tmp_path) -> None:
    from services.api.app.storage import S3ArtifactStorage

    source = tmp_path / "result.zip"
    source.write_bytes(b"abcdefgh")
    client = FakeS3Client()
    storage = S3ArtifactStorage(
        "ocr-results",
        prefix="lao-ocr/prod",
        endpoint_url="https://example.invalid",
        region_name="auto",
        force_path_style=True,
        client=client,
    )

    artifact = storage.put_file(
        source,
        key="jobs/job-1/result.zip",
        filename="result.zip",
        media_type="application/zip",
    )

    object_key = "lao-ocr/prod/jobs/job-1/result.zip"
    assert ("ocr-results", object_key) in client.objects
    assert client.objects[("ocr-results", object_key)]["extra"] == {
        "ContentType": "application/zip",
    }
    assert artifact.size_bytes == 8
    assert storage.exists(artifact) is True
    assert b"".join(storage.iter_bytes(artifact, chunk_size=3)) == b"abcdefgh"
    assert client.last_body is not None and client.last_body.closed is True

    storage.delete(artifact)
    assert storage.exists(artifact) is False


def test_s3_storage_missing_object_returns_false(tmp_path) -> None:
    from services.api.app.storage import S3ArtifactStorage, StoredArtifact

    storage = S3ArtifactStorage("bucket", client=FakeS3Client())
    artifact = StoredArtifact(
        key="jobs/missing.zip",
        filename="missing.zip",
        media_type="application/zip",
        size_bytes=1,
    )

    assert storage.exists(artifact) is False


def test_s3_storage_rejects_unsafe_prefix_and_keys(tmp_path) -> None:
    from services.api.app.storage import S3ArtifactStorage

    with pytest.raises(ValueError, match="storage key"):
        S3ArtifactStorage("bucket", prefix="../escape", client=FakeS3Client())

    source = tmp_path / "source.bin"
    source.write_bytes(b"x")
    storage = S3ArtifactStorage("bucket", client=FakeS3Client())
    with pytest.raises(ValueError, match="storage key"):
        storage.put_file(
            source,
            key="../escape.bin",
            filename="escape.bin",
            media_type="application/octet-stream",
        )


def test_s3_storage_metadata_hides_credentials() -> None:
    from services.api.app.storage import S3ArtifactStorage

    storage = S3ArtifactStorage(
        "bucket",
        prefix="results",
        endpoint_url="https://r2.example",
        region_name="auto",
        force_path_style=False,
        client=FakeS3Client(),
    )

    assert storage.metadata() == {
        "backend": "s3",
        "bucket": "bucket",
        "prefix": "results",
        "endpoint_url": "https://r2.example",
        "region_name": "auto",
        "force_path_style": False,
    }
