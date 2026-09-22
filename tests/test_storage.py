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
