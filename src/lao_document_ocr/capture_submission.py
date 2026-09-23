from __future__ import annotations

import hashlib
import io
import json
import os
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from PIL import Image

from lao_document_ocr.capture_registration import CaptureMode

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CAPTURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
_MAX_SUBMISSION_BYTES = 2 * 1024 * 1024 * 1024
_MAX_UNCOMPRESSED_BYTES = 3 * 1024 * 1024 * 1024
_MAX_MEMBERS = 512
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


class CaptureSubmissionError(ValueError):
    pass


@dataclass(frozen=True)
class CaptureSubmissionFile:
    page_id: str
    filename: str
    member: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CaptureSubmission:
    schema_version: str
    suite_id: str
    source_revision: str | None
    kit_sha256: str
    capture_id: str
    mode: str
    require_qr: bool
    expected_pages: int
    captured_pages: int
    complete: bool
    files: tuple[CaptureSubmissionFile, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["files"] = [item.to_dict() for item in self.files]
        return payload


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in name
    )


def _zip_info(name: str, *, mode: int = 0o600) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = mode << 16
    info.create_system = 3
    return info


def _session_metadata(capture_dir: Path) -> dict:
    path = capture_dir / ".collector-session.json"
    if not path.is_file():
        raise CaptureSubmissionError(
            "Capture directory is missing .collector-session.json."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureSubmissionError(
            "Collector session metadata is invalid."
        ) from exc

    if payload.get("schema_version") != "1":
        raise CaptureSubmissionError("Unsupported collector session schema.")

    for key in ("suite_id", "capture_id"):
        value = str(payload.get(key, "")).strip()
        if not _SAFE_ID.fullmatch(value):
            raise CaptureSubmissionError(
                f"Collector session has invalid {key}."
            )

    mode = str(payload.get("mode", ""))
    if mode not in {item.value for item in CaptureMode}:
        raise CaptureSubmissionError("Collector session has invalid mode.")

    kit_sha256 = str(payload.get("kit_sha256", "")).lower()
    if not _HEX64.fullmatch(kit_sha256):
        raise CaptureSubmissionError(
            "Collector session has invalid kit_sha256."
        )

    expected = payload.get("expected_pages")
    if not isinstance(expected, int) or expected < 1:
        raise CaptureSubmissionError(
            "Collector session has invalid expected_pages."
        )
    if not isinstance(payload.get("require_qr"), bool):
        raise CaptureSubmissionError(
            "Collector session has invalid require_qr."
        )
    return payload


def _validate_image(path: Path, *, max_page_pixels: int) -> None:
    try:
        with Image.open(path) as image:
            if image.width < 1 or image.height < 1:
                raise CaptureSubmissionError(
                    f"{path.name}: invalid image dimensions."
                )
            if image.width * image.height > max_page_pixels:
                raise CaptureSubmissionError(
                    f"{path.name}: image dimensions exceed the pixel limit."
                )
            image.verify()
    except CaptureSubmissionError:
        raise
    except Exception as exc:
        raise CaptureSubmissionError(
            f"{path.name}: invalid capture image."
        ) from exc


def _capture_files(
    capture_dir: Path,
    *,
    max_page_pixels: int,
) -> tuple[CaptureSubmissionFile, ...]:
    items: list[CaptureSubmissionFile] = []
    seen_page_ids: set[str] = set()

    for path in sorted(capture_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.suffix.lower() not in _CAPTURE_EXTENSIONS:
            continue
        page_id = path.stem
        if not _SAFE_ID.fullmatch(page_id):
            raise CaptureSubmissionError(
                f"Unsafe capture page id in filename: {path.name}"
            )
        if page_id in seen_page_ids:
            raise CaptureSubmissionError(
                f"Multiple capture files exist for page id: {page_id}"
            )
        _validate_image(path, max_page_pixels=max_page_pixels)
        data = path.read_bytes()
        items.append(
            CaptureSubmissionFile(
                page_id=page_id,
                filename=path.name,
                member=f"captures/{path.name}",
                size_bytes=len(data),
                sha256=_sha256_bytes(data),
            )
        )
        seen_page_ids.add(page_id)

    if not items:
        raise CaptureSubmissionError(
            "Capture directory contains no capture images."
        )
    return tuple(items)


def build_capture_submission(
    capture_dir: str | Path,
    output_path: str | Path,
    *,
    require_complete: bool = False,
    max_page_pixels: int = 80_000_000,
) -> Path:
    root = Path(capture_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Capture directory not found: {root}")
    if max_page_pixels < 1:
        raise ValueError("max_page_pixels must be at least 1")

    metadata = _session_metadata(root)
    files = _capture_files(
        root,
        max_page_pixels=max_page_pixels,
    )
    expected = int(metadata["expected_pages"])
    if len(files) > expected:
        raise CaptureSubmissionError(
            "Capture directory has more images than expected pages."
        )
    if require_complete and len(files) != expected:
        raise CaptureSubmissionError(
            f"Capture session is incomplete ({len(files)}/{expected})."
        )

    submission = CaptureSubmission(
        schema_version="1",
        suite_id=str(metadata["suite_id"]),
        source_revision=(
            str(metadata["source_revision"])
            if metadata.get("source_revision") is not None
            else None
        ),
        kit_sha256=str(metadata["kit_sha256"]),
        capture_id=str(metadata["capture_id"]),
        mode=str(metadata["mode"]),
        require_qr=bool(metadata["require_qr"]),
        expected_pages=expected,
        captured_pages=len(files),
        complete=len(files) == expected,
        files=files,
    )
    manifest_bytes = (
        json.dumps(
            submission.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")

    payloads: dict[str, bytes] = {
        "capture-submission.json": manifest_bytes,
    }
    for item in files:
        payloads[item.member] = (root / item.filename).read_bytes()

    sums = "\n".join(
        f"{_sha256_bytes(data)}  {name}"
        for name, data in sorted(payloads.items())
    ) + "\n"
    payloads["SHA256SUMS"] = sums.encode("utf-8")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        with zipfile.ZipFile(
            temp,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name, data in sorted(payloads.items()):
                archive.writestr(_zip_info(name), data)
        os.replace(temp, destination)
        try:
            destination.chmod(0o600)
        except OSError:
            pass
    finally:
        temp.unlink(missing_ok=True)
    return destination


def _parse_sums(data: bytes) -> dict[str, str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CaptureSubmissionError(
            "Submission SHA256SUMS is not UTF-8."
        ) from exc

    output: dict[str, str] = {}
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise CaptureSubmissionError(
                f"Invalid SHA256SUMS entry at line {line_number}."
            )
        digest, name = parts[0].lower(), parts[1].strip()
        if not _HEX64.fullmatch(digest):
            raise CaptureSubmissionError(
                f"Invalid SHA-256 digest at line {line_number}."
            )
        if not _safe_member(name):
            raise CaptureSubmissionError(
                f"Unsafe SHA256SUMS member at line {line_number}."
            )
        if name in output:
            raise CaptureSubmissionError(
                f"Duplicate SHA256SUMS entry: {name}"
            )
        output[name] = digest
    return output


def verify_capture_submission(
    submission_path: str | Path,
    *,
    max_page_pixels: int = 80_000_000,
) -> CaptureSubmission:
    source = Path(submission_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Capture submission not found: {source}")
    if source.stat().st_size > _MAX_SUBMISSION_BYTES:
        raise CaptureSubmissionError(
            "Capture submission ZIP exceeds the size limit."
        )

    try:
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if len(names) > _MAX_MEMBERS:
                raise CaptureSubmissionError(
                    "Capture submission contains too many ZIP members."
                )
            if sum(item.file_size for item in infos) > _MAX_UNCOMPRESSED_BYTES:
                raise CaptureSubmissionError(
                    "Capture submission uncompressed size exceeds the limit."
                )
            if len(names) != len(set(names)):
                raise CaptureSubmissionError(
                    "Capture submission contains duplicate ZIP members."
                )
            unsafe = [name for name in names if not _safe_member(name)]
            if unsafe:
                raise CaptureSubmissionError(
                    f"Unsafe capture submission member: {unsafe[0]}"
                )
            required = {"capture-submission.json", "SHA256SUMS"}
            missing = sorted(required - set(names))
            if missing:
                raise CaptureSubmissionError(
                    "Capture submission is missing required files: "
                    + ", ".join(missing)
                )

            sums = _parse_sums(archive.read("SHA256SUMS"))
            expected = set(names) - {"SHA256SUMS"}
            if set(sums) != expected:
                raise CaptureSubmissionError(
                    "Capture submission checksum coverage does not match ZIP contents."
                )
            for name, digest in sums.items():
                actual = _sha256_bytes(archive.read(name))
                if actual != digest:
                    raise CaptureSubmissionError(
                        f"Capture submission checksum mismatch: {name}"
                    )

            payload = json.loads(
                archive.read("capture-submission.json").decode("utf-8")
            )
            if payload.get("schema_version") != "1":
                raise CaptureSubmissionError(
                    "Unsupported capture submission schema."
                )
            files_raw = payload.get("files")
            if not isinstance(files_raw, list) or not files_raw:
                raise CaptureSubmissionError(
                    "Capture submission contains no file records."
                )

            files: list[CaptureSubmissionFile] = []
            members_from_manifest: set[str] = set()
            page_ids: set[str] = set()
            for item in files_raw:
                if not isinstance(item, dict):
                    raise CaptureSubmissionError(
                        "Invalid capture submission file record."
                    )
                page_id = str(item.get("page_id", ""))
                filename = str(item.get("filename", ""))
                member = str(item.get("member", ""))
                digest = str(item.get("sha256", "")).lower()
                size = item.get("size_bytes")
                if not _SAFE_ID.fullmatch(page_id):
                    raise CaptureSubmissionError(
                        "Capture submission has an unsafe page_id."
                    )
                if page_id in page_ids:
                    raise CaptureSubmissionError(
                        f"Duplicate capture page id: {page_id}"
                    )
                if (
                    not _safe_member(member)
                    or member != f"captures/{filename}"
                    or Path(filename).name != filename
                    or Path(filename).stem != page_id
                    or Path(filename).suffix.lower() not in _CAPTURE_EXTENSIONS
                ):
                    raise CaptureSubmissionError(
                        f"Invalid capture member for page {page_id}."
                    )
                if member not in names:
                    raise CaptureSubmissionError(
                        f"Missing capture member: {member}"
                    )
                if not _HEX64.fullmatch(digest):
                    raise CaptureSubmissionError(
                        f"Invalid capture hash for page {page_id}."
                    )
                data = archive.read(member)
                if len(data) != size or _sha256_bytes(data) != digest:
                    raise CaptureSubmissionError(
                        f"Capture metadata mismatch for page {page_id}."
                    )
                with Image.open(io.BytesIO(data)) as image:
                    if image.width * image.height > max_page_pixels:
                        raise CaptureSubmissionError(
                            f"{filename}: image dimensions exceed the pixel limit."
                        )
                    image.verify()
                files.append(
                    CaptureSubmissionFile(
                        page_id=page_id,
                        filename=filename,
                        member=member,
                        size_bytes=int(size),
                        sha256=digest,
                    )
                )
                page_ids.add(page_id)
                members_from_manifest.add(member)

            capture_members = {
                name for name in names if name.startswith("captures/")
            }
            if members_from_manifest != capture_members:
                raise CaptureSubmissionError(
                    "Capture submission file records do not match ZIP capture members."
                )

            suite_id = str(payload.get("suite_id", ""))
            capture_id = str(payload.get("capture_id", ""))
            mode = str(payload.get("mode", ""))
            kit_sha256 = str(payload.get("kit_sha256", "")).lower()
            expected_pages = payload.get("expected_pages")
            captured_pages = payload.get("captured_pages")
            complete = payload.get("complete")
            require_qr = payload.get("require_qr")
            if not _SAFE_ID.fullmatch(suite_id):
                raise CaptureSubmissionError("Invalid submission suite_id.")
            if not _SAFE_ID.fullmatch(capture_id):
                raise CaptureSubmissionError("Invalid submission capture_id.")
            if mode not in {item.value for item in CaptureMode}:
                raise CaptureSubmissionError("Invalid submission capture mode.")
            if not _HEX64.fullmatch(kit_sha256):
                raise CaptureSubmissionError("Invalid submission kit_sha256.")
            if (
                not isinstance(expected_pages, int)
                or expected_pages < 1
                or not isinstance(captured_pages, int)
                or captured_pages != len(files)
                or captured_pages > expected_pages
            ):
                raise CaptureSubmissionError(
                    "Invalid submission page counts."
                )
            if complete is not (captured_pages == expected_pages):
                raise CaptureSubmissionError(
                    "Invalid submission completeness flag."
                )
            if not isinstance(require_qr, bool):
                raise CaptureSubmissionError(
                    "Invalid submission require_qr flag."
                )

            return CaptureSubmission(
                schema_version="1",
                suite_id=suite_id,
                source_revision=(
                    str(payload["source_revision"])
                    if payload.get("source_revision") is not None
                    else None
                ),
                kit_sha256=kit_sha256,
                capture_id=capture_id,
                mode=mode,
                require_qr=require_qr,
                expected_pages=expected_pages,
                captured_pages=captured_pages,
                complete=complete,
                files=tuple(files),
            )
    except zipfile.BadZipFile as exc:
        raise CaptureSubmissionError(
            "Invalid capture submission ZIP."
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CaptureSubmissionError(
            "Invalid capture submission metadata."
        ) from exc
    except CaptureSubmissionError:
        raise
    except Exception as exc:
        raise CaptureSubmissionError(
            "Invalid capture submission."
        ) from exc


def extract_capture_submission(
    submission_path: str | Path,
    output_dir: str | Path,
    *,
    max_page_pixels: int = 80_000_000,
) -> Path:
    submission = verify_capture_submission(
        submission_path,
        max_page_pixels=max_page_pixels,
    )
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        destination.chmod(0o700)
    except OSError:
        pass

    occupied = [
        item for item in destination.iterdir()
        if item.is_file()
    ]
    if occupied:
        raise CaptureSubmissionError(
            "Extraction directory must be empty."
        )

    with zipfile.ZipFile(submission_path) as archive:
        for item in submission.files:
            data = archive.read(item.member)
            path = destination / item.filename
            path.write_bytes(data)
            try:
                path.chmod(0o600)
            except OSError:
                pass

    metadata = destination / "capture-submission.json"
    metadata.write_text(
        json.dumps(
            submission.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        metadata.chmod(0o600)
    except OSError:
        pass
    return destination
