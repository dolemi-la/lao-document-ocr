# ruff: noqa: E501
from __future__ import annotations

import csv
import hashlib
import hmac
import html
import io
import json
import os
import re
import threading
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image

from lao_document_ocr.capture_page_id import decode_page_id
from lao_document_ocr.capture_registration import CaptureMode

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_ACCESS_TOKEN = re.compile(r"^[A-Za-z0-9._~-]{16,256}$")
_COLLECTOR_COOKIE = "lao_ocr_collector"
_ALLOWED_IMAGE_SUFFIXES = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".webp": "WEBP",
}
_REQUIRED_KIT_MEMBERS = {
    "capture-kit.json",
    "capture-worksheet.csv",
    "SHA256SUMS",
}
_MAX_KIT_BYTES = 100 * 1024 * 1024
_MAX_KIT_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
_MAX_KIT_MEMBERS = 32


class CaptureCollectorError(ValueError):
    pass


@dataclass(frozen=True)
class CollectorPage:
    combined_page: int
    template: str
    page_id: str
    required_modes: tuple[str, ...]


@dataclass(frozen=True)
class CollectorKit:
    path: Path
    suite_id: str
    source_revision: str | None
    pdf_name: str
    pdf_bytes: bytes
    pages: tuple[CollectorPage, ...]
    sha256: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in name
    )


def _parse_sha256sums(data: bytes) -> dict[str, str]:
    checksums: dict[str, str] = {}
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CaptureCollectorError("Capture kit SHA256SUMS is not UTF-8.") from exc

    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise CaptureCollectorError(
                f"Invalid SHA256SUMS entry at line {line_number}."
            )
        digest, name = parts[0].lower(), parts[1].strip()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise CaptureCollectorError(
                f"Invalid SHA-256 digest at line {line_number}."
            )
        if not _safe_member_name(name):
            raise CaptureCollectorError(
                f"Unsafe SHA256SUMS member at line {line_number}."
            )
        if name in checksums:
            raise CaptureCollectorError(f"Duplicate checksum entry: {name}")
        checksums[name] = digest
    if not checksums:
        raise CaptureCollectorError("Capture kit SHA256SUMS is empty.")
    return checksums


def _parse_worksheet(data: bytes) -> tuple[CollectorPage, ...]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CaptureCollectorError("Capture worksheet is not UTF-8.") from exc

    reader = csv.DictReader(io.StringIO(text))
    required = {
        "combined_page",
        "template",
        "page_id",
        "required_capture_modes",
    }
    fields = set(reader.fieldnames or [])
    missing = sorted(required - fields)
    if missing:
        raise CaptureCollectorError(
            "Capture worksheet is missing required columns: "
            + ", ".join(missing)
        )

    allowed_modes = {mode.value for mode in CaptureMode}
    pages: list[CollectorPage] = []
    seen_ids: set[str] = set()
    seen_numbers: set[int] = set()

    for row_number, row in enumerate(reader, start=2):
        try:
            combined_page = int(row["combined_page"])
        except (TypeError, ValueError) as exc:
            raise CaptureCollectorError(
                f"Invalid combined_page at worksheet row {row_number}."
            ) from exc
        if combined_page < 1 or combined_page in seen_numbers:
            raise CaptureCollectorError(
                f"Duplicate/invalid combined page at worksheet row {row_number}."
            )

        page_id = (row.get("page_id") or "").strip()
        template = (row.get("template") or "").strip()
        if not _SAFE_ID.fullmatch(page_id):
            raise CaptureCollectorError(
                f"Unsafe page_id at worksheet row {row_number}."
            )
        if page_id in seen_ids:
            raise CaptureCollectorError(f"Duplicate page_id: {page_id}")
        if not template:
            raise CaptureCollectorError(
                f"Missing template at worksheet row {row_number}."
            )

        modes = tuple(
            sorted(
                {
                    value.strip()
                    for value in (row.get("required_capture_modes") or "").split(";")
                    if value.strip()
                }
            )
        )
        if not modes:
            raise CaptureCollectorError(
                f"Page {page_id} has no required capture modes."
            )
        unknown = sorted(set(modes) - allowed_modes)
        if unknown:
            raise CaptureCollectorError(
                f"Page {page_id} has unsupported capture modes: {', '.join(unknown)}"
            )

        pages.append(
            CollectorPage(
                combined_page=combined_page,
                template=template,
                page_id=page_id,
                required_modes=modes,
            )
        )
        seen_ids.add(page_id)
        seen_numbers.add(combined_page)

    if not pages:
        raise CaptureCollectorError("Capture worksheet contains no pages.")
    return tuple(sorted(pages, key=lambda page: page.combined_page))


def load_collector_kit(path: str | Path) -> CollectorKit:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Collector capture kit not found: {source}")
    if source.stat().st_size > _MAX_KIT_BYTES:
        raise CaptureCollectorError("Collector capture-kit ZIP exceeds the size limit.")

    try:
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) > _MAX_KIT_MEMBERS:
                raise CaptureCollectorError("Capture kit contains too many ZIP members.")
            if sum(info.file_size for info in infos) > _MAX_KIT_UNCOMPRESSED_BYTES:
                raise CaptureCollectorError(
                    "Capture kit uncompressed size exceeds the limit."
                )
            if len(names) != len(set(names)):
                raise CaptureCollectorError("Capture kit contains duplicate ZIP members.")
            unsafe = [name for name in names if not _safe_member_name(name)]
            if unsafe:
                raise CaptureCollectorError(
                    f"Capture kit contains unsafe ZIP member: {unsafe[0]}"
                )
            missing = sorted(_REQUIRED_KIT_MEMBERS - set(names))
            if missing:
                raise CaptureCollectorError(
                    "Capture kit is missing required files: " + ", ".join(missing)
                )

            checksums = _parse_sha256sums(archive.read("SHA256SUMS"))
            expected_members = set(names) - {"SHA256SUMS"}
            if set(checksums) != expected_members:
                raise CaptureCollectorError(
                    "Capture kit checksum coverage does not match ZIP contents."
                )
            for name, expected in checksums.items():
                actual = _sha256_bytes(archive.read(name))
                if actual != expected:
                    raise CaptureCollectorError(
                        f"Capture-kit checksum mismatch: {name}"
                    )

            kit_manifest = json.loads(
                archive.read("capture-kit.json").decode("utf-8")
            )
            if kit_manifest.get("schema_version") != "1":
                raise CaptureCollectorError(
                    "Unsupported capture-kit schema version."
                )
            if not all(
                kit_manifest.get(flag) is True
                for flag in (
                    "excludes_ground_truth",
                    "excludes_digital_page_images",
                    "excludes_internal_suite_paths",
                )
            ):
                raise CaptureCollectorError(
                    "Collector kit does not satisfy blind-collection exclusions."
                )

            suite_id = str(kit_manifest.get("suite_id", "")).strip()
            if not _SAFE_ID.fullmatch(suite_id):
                raise CaptureCollectorError("Capture kit has an unsafe suite_id.")

            pdf_name = f"{suite_id}.pdf"
            if pdf_name not in checksums:
                raise CaptureCollectorError(
                    f"Capture kit is missing checksummed printable PDF: {pdf_name}"
                )
            pages = _parse_worksheet(archive.read("capture-worksheet.csv"))
            if int(kit_manifest.get("page_count", -1)) != len(pages):
                raise CaptureCollectorError(
                    "Capture-kit page_count does not match worksheet."
                )

            return CollectorKit(
                path=source,
                suite_id=suite_id,
                source_revision=kit_manifest.get("source_revision"),
                pdf_name=pdf_name,
                pdf_bytes=archive.read(pdf_name),
                pages=pages,
                sha256=_sha256_bytes(source.read_bytes()),
            )
    except zipfile.BadZipFile as exc:
        raise CaptureCollectorError("Invalid collector capture-kit ZIP.") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CaptureCollectorError("Invalid collector capture-kit metadata.") from exc


class CollectorSession:
    def __init__(
        self,
        kit: CollectorKit,
        output_dir: str | Path,
        *,
        capture_id: str,
        mode: CaptureMode,
        max_upload_bytes: int = 25 * 1024 * 1024,
        max_page_pixels: int = 80_000_000,
        require_qr: bool = True,
    ) -> None:
        if not _SAFE_ID.fullmatch(capture_id):
            raise CaptureCollectorError(
                "capture_id may contain only letters, numbers, '.', '_' and '-'."
            )
        if max_upload_bytes < 1:
            raise ValueError("max_upload_bytes must be at least 1")
        if max_page_pixels < 1:
            raise ValueError("max_page_pixels must be at least 1")

        self.kit = kit
        self.capture_id = capture_id
        self.mode = mode
        self.max_upload_bytes = max_upload_bytes
        self.max_page_pixels = max_page_pixels
        self.require_qr = require_qr
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.output_dir.chmod(0o700)
        self.pages = tuple(
            page for page in kit.pages if mode.value in page.required_modes
        )
        if not self.pages:
            raise CaptureCollectorError(
                f"Capture kit has no pages requiring mode {mode.value}."
            )
        self.page_map = {page.page_id: page for page in self.pages}
        self._lock = threading.RLock()
        self._write_session_metadata()

    def _write_session_metadata(self) -> None:
        path = self.output_dir / ".collector-session.json"
        payload = {
            "schema_version": "1",
            "suite_id": self.kit.suite_id,
            "source_revision": self.kit.source_revision,
            "kit_sha256": self.kit.sha256,
            "capture_id": self.capture_id,
            "mode": self.mode.value,
            "require_qr": self.require_qr,
            "expected_pages": len(self.pages),
        }
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CaptureCollectorError(
                    "Existing collector session metadata is invalid."
                ) from exc
            comparable_keys = (
                "schema_version",
                "suite_id",
                "kit_sha256",
                "capture_id",
                "mode",
                "require_qr",
                "expected_pages",
            )
            if any(existing.get(key) != payload.get(key) for key in comparable_keys):
                raise CaptureCollectorError(
                    "Output directory belongs to a different collector session."
                )
            path.chmod(0o600)
            return

        existing_images = [
            item
            for item in self.output_dir.iterdir()
            if item.is_file() and item.suffix.lower() in _ALLOWED_IMAGE_SUFFIXES
        ]
        if existing_images:
            raise CaptureCollectorError(
                "Output directory already contains captures but has no collector "
                "session metadata. Use a new/empty directory."
            )

        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        path.chmod(0o600)

    def existing_files(self, page_id: str) -> list[Path]:
        return sorted(
            path
            for suffix in _ALLOWED_IMAGE_SUFFIXES
            for path in [self.output_dir / f"{page_id}{suffix}"]
            if path.is_file()
        )

    def progress(self) -> dict:
        pages = []
        completed = 0
        for page in self.pages:
            existing = self.existing_files(page.page_id)
            if len(existing) > 1:
                state = "duplicate"
            elif existing:
                state = "complete"
                completed += 1
            else:
                state = "pending"
            pages.append(
                {
                    "combined_page": page.combined_page,
                    "template": page.template,
                    "page_id": page.page_id,
                    "required_modes": list(page.required_modes),
                    "state": state,
                    "capture_file": existing[0].name if len(existing) == 1 else None,
                }
            )
        return {
            "suite_id": self.kit.suite_id,
            "source_revision": self.kit.source_revision,
            "kit_sha256": self.kit.sha256,
            "capture_id": self.capture_id,
            "mode": self.mode.value,
            "require_qr": self.require_qr,
            "expected": len(self.pages),
            "completed": completed,
            "remaining": len(self.pages) - completed,
            "pages": pages,
        }

    def page(self, page_id: str) -> CollectorPage:
        page = self.page_map.get(page_id)
        if page is None:
            raise KeyError(page_id)
        return page

    def commit_capture(
        self,
        page_id: str,
        temp_path: Path,
        suffix: str,
    ) -> Path:
        self.page(page_id)
        destination = self.output_dir / f"{page_id}{suffix}"
        with self._lock:
            if self.existing_files(page_id):
                raise FileExistsError(page_id)
            os.replace(temp_path, destination)
            os.chmod(destination, 0o600)
        return destination

    def delete(self, page_id: str) -> int:
        self.page(page_id)
        with self._lock:
            files = self.existing_files(page_id)
            for path in files:
                path.unlink(missing_ok=True)
        return len(files)


def _validate_capture_image(
    path: Path,
    suffix: str,
    *,
    max_page_pixels: int,
) -> None:
    expected = _ALLOWED_IMAGE_SUFFIXES.get(suffix)
    if expected is None:
        raise CaptureCollectorError("Unsupported capture image type.")
    try:
        with Image.open(path) as image:
            actual = (image.format or "").upper()
            if actual != expected:
                raise CaptureCollectorError(
                    f"File content does not match the {suffix} extension."
                )
            if image.width < 1 or image.height < 1:
                raise CaptureCollectorError("Capture image has invalid dimensions.")
            if image.width * image.height > max_page_pixels:
                raise CaptureCollectorError(
                    "Capture image dimensions exceed the pixel limit."
                )
            image.verify()
    except CaptureCollectorError:
        raise
    except Exception as exc:
        raise CaptureCollectorError("Invalid capture image.") from exc


async def _save_upload(
    upload: UploadFile,
    destination: Path,
    *,
    max_upload_bytes: int,
) -> int:
    total = 0
    with destination.open("xb") as handle:
        os.chmod(destination, 0o600)
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > max_upload_bytes:
                raise CaptureCollectorError(
                    f"Capture exceeds the {max_upload_bytes} byte upload limit."
                )
            handle.write(chunk)
    if total == 0:
        raise CaptureCollectorError("Capture upload is empty.")
    return total


def _collector_html(session: CollectorSession) -> str:
    suite = html.escape(session.kit.suite_id)
    mode = html.escape(session.mode.value)
    capture_id = html.escape(session.capture_id)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lao OCR Collector · {suite}</title>
<style>
:root{{font-family:Inter,system-ui,sans-serif;color:#17171c;background:#f6f5fb}}
*{{box-sizing:border-box}}
body{{margin:0;padding:18px}}
main{{max-width:720px;margin:auto}}
.card{{background:white;border:1px solid #ddd9ef;border-radius:18px;padding:18px;box-shadow:0 8px 24px #2f23500d}}
h1{{margin:0 0 8px;font-size:24px}}
.meta{{color:#666;margin-bottom:16px}}
.progress{{height:10px;background:#eceaf5;border-radius:99px;overflow:hidden}}
.progress>div{{height:100%;background:#6d5bd0;width:0}}
select,input,button,a.button{{width:100%;font:inherit;padding:13px;border-radius:12px;margin-top:10px}}
select,input{{border:1px solid #c9c4dc;background:white}}
button,a.button{{border:0;background:#6d5bd0;color:white;font-weight:700;text-align:center;text-decoration:none;display:block}}
button.secondary{{background:#eeeafc;color:#4b3e9e}}
button.danger{{background:#fbe9e9;color:#9d2d2d}}
#message{{min-height:24px;margin-top:12px;color:#51486f}}
small{{color:#777;display:block;margin-top:8px}}
</style>
</head>
<body>
<main>
<div class="card">
<h1>Lao OCR capture collector</h1>
<div class="meta">Suite <strong>{suite}</strong> · mode <strong>{mode}</strong> · session <strong>{capture_id}</strong></div>
<div id="count">Loading progress…</div>
<div class="progress"><div id="bar"></div></div>
<label><small>Page to capture</small><select id="page"></select></label>
<label><small>Camera / image file</small><input id="file" type="file" accept="image/*" capture="environment"></label>
<button id="upload">Save capture</button>
<button id="retake" class="danger" type="button">Delete saved capture / retake</button>
<a class="button secondary" href="/kit.pdf" target="_blank" rel="noopener">Open printable reference PDF</a>
<div id="message"></div>
<small>Use only on a trusted local network. This collector kit contains no ground truth.</small>
</div>
</main>
<script>
let session=null;
async function refresh(){{
  const response=await fetch('/api/session',{{cache:'no-store'}});
  session=await response.json();
  document.getElementById('count').textContent=session.completed+' / '+session.expected+' captured · '+session.remaining+' remaining';
  document.getElementById('bar').style.width=(session.expected?100*session.completed/session.expected:0)+'%';
  const select=document.getElementById('page');
  const selected=select.value;
  select.innerHTML='';
  for(const page of session.pages){{
    const option=document.createElement('option');
    option.value=page.page_id;
    option.textContent=page.combined_page+' · '+page.template+' · '+page.page_id+' · '+page.state;
    select.appendChild(option);
  }}
  if(Array.from(select.options).some(option=>option.value===selected)) select.value=selected;
  else {{
    const pending=session.pages.find(page=>page.state==='pending');
    if(pending) select.value=pending.page_id;
  }}
}}
document.getElementById('upload').onclick=async()=>{{
  const page=document.getElementById('page').value;
  const file=document.getElementById('file').files[0];
  if(!page||!file){{document.getElementById('message').textContent='Choose a page and capture first.';return;}}
  const form=new FormData(); form.append('file',file);
  const response=await fetch('/api/captures/'+encodeURIComponent(page),{{method:'POST',body:form}});
  const payload=await response.json().catch(()=>({{}}));
  document.getElementById('message').textContent=response.ok?'Saved '+payload.capture_file:(payload.detail||'Upload failed');
  if(response.ok) document.getElementById('file').value='';
  await refresh();
}};
document.getElementById('retake').onclick=async()=>{{
  const page=document.getElementById('page').value;
  if(!page)return;
  const response=await fetch('/api/captures/'+encodeURIComponent(page),{{method:'DELETE'}});
  const payload=await response.json().catch(()=>({{}}));
  document.getElementById('message').textContent=response.ok?'Deleted '+payload.deleted+' capture file(s)':(payload.detail||'Delete failed');
  await refresh();
}};
refresh();
</script>
</body>
</html>"""


def create_collector_app(
    kit_path: str | Path,
    output_dir: str | Path,
    *,
    capture_id: str,
    mode: CaptureMode,
    max_upload_bytes: int = 25 * 1024 * 1024,
    max_page_pixels: int = 80_000_000,
    require_qr: bool = True,
    access_token: str | None = None,
) -> FastAPI:
    if access_token is not None and not _ACCESS_TOKEN.fullmatch(access_token):
        raise CaptureCollectorError(
            "access_token must be 16-256 URL-safe characters."
        )

    kit = load_collector_kit(kit_path)
    session = CollectorSession(
        kit,
        output_dir,
        capture_id=capture_id,
        mode=mode,
        max_upload_bytes=max_upload_bytes,
        max_page_pixels=max_page_pixels,
        require_qr=require_qr,
    )
    app = FastAPI(
        title="Lao OCR Capture Collector",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        authenticated_by_query = False
        if access_token is not None:
            supplied = (
                request.headers.get("x-collector-token")
                or request.query_params.get("token")
                or request.cookies.get(_COLLECTOR_COOKIE)
            )
            if supplied is None or not hmac.compare_digest(
                supplied,
                access_token,
            ):
                response = JSONResponse(
                    status_code=403,
                    content={"detail": "Collector access token required."},
                )
            else:
                authenticated_by_query = (
                    request.query_params.get("token") is not None
                )
                response = await call_next(request)
        else:
            response = await call_next(request)

        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'unsafe-inline'; "
            "style-src 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        )
        if access_token is not None and authenticated_by_query:
            response.set_cookie(
                _COLLECTOR_COOKIE,
                access_token,
                httponly=True,
                samesite="strict",
                secure=False,
                path="/",
            )
        return response

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _collector_html(session)

    @app.get("/api/session")
    def get_session() -> dict:
        return session.progress()

    @app.get("/kit.pdf")
    def kit_pdf() -> Response:
        return Response(
            content=kit.pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="{kit.pdf_name}"',
            },
        )

    @app.post("/api/captures/{page_id}", status_code=201)
    async def upload_capture(
        page_id: str,
        file: Annotated[UploadFile, File(...)],
    ) -> dict:
        try:
            session.page(page_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown page ID.") from exc

        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in _ALLOWED_IMAGE_SUFFIXES:
            raise HTTPException(
                status_code=422,
                detail="Capture must be PNG, JPG, TIFF, or WebP.",
            )
        existing = session.existing_files(page_id)
        if existing:
            raise HTTPException(
                status_code=409,
                detail="A capture already exists for this page. Delete it before retaking.",
            )

        temp_path = session.output_dir / f".upload-{uuid.uuid4().hex}{suffix}"
        try:
            await _save_upload(
                file,
                temp_path,
                max_upload_bytes=session.max_upload_bytes,
            )
            _validate_capture_image(
                temp_path,
                suffix,
                max_page_pixels=session.max_page_pixels,
            )
            decoded_page_id = decode_page_id(temp_path)
            if decoded_page_id is None and session.require_qr:
                raise CaptureCollectorError(
                    "Could not read the printed page-ID QR marker from this capture."
                )
            if decoded_page_id is not None and decoded_page_id != page_id:
                raise CaptureCollectorError(
                    "Capture QR page ID does not match the selected page "
                    f"({decoded_page_id} != {page_id})."
                )
            try:
                destination = session.commit_capture(
                    page_id,
                    temp_path,
                    suffix,
                )
            except FileExistsError as exc:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "A capture already exists for this page. "
                        "Delete it before retaking."
                    ),
                ) from exc
        except CaptureCollectorError as exc:
            temp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

        return {
            "page_id": page_id,
            "capture_file": destination.name,
            "size_bytes": destination.stat().st_size,
        }

    @app.delete("/api/captures/{page_id}")
    def delete_capture(page_id: str) -> dict:
        try:
            deleted = session.delete(page_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown page ID.") from exc
        return {"page_id": page_id, "deleted": deleted}

    return app
