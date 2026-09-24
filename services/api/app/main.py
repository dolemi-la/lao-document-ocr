from __future__ import annotations

import io
import os
import tempfile
import uuid
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse

from lao_document_ocr.exporters import (
    export_docx,
    export_json,
    export_markdown,
    export_text,
)
from lao_document_ocr.ocr import (
    OcrEngine,
    OcrEngineError,
    OwnedRecognizerEngine,
    TesseractEngine,
)
from lao_document_ocr.pipeline import (
    SUPPORTED_SUFFIXES,
    DocumentProcessingCancelled,
    DocumentProcessingError,
    process_document,
)
from lao_document_ocr.reading_order import DeterministicReadingOrderResolver
from services.api.app.jobs import (
    ConversionJobManager,
    JobCancelledError,
    JobCapacityError,
    JobNotFoundError,
    JobPublicError,
    JobRecord,
    JobStatus,
)
from services.api.app.metrics import ApiMetrics, RequestTimer
from services.api.app.rate_limit import SlidingWindowRateLimiter
from services.api.app.security import (
    UploadValidationError,
    sanitize_filename,
    validate_uploaded_content,
)
from services.api.app.storage import (
    FilesystemArtifactStorage,
    S3ArtifactStorage,
    StoredArtifact,
)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_PAGES = int(os.getenv("MAX_PAGES", "60"))
MAX_PAGE_PIXELS = int(os.getenv("MAX_PAGE_PIXELS", "40000000"))
OCR_ENGINE = os.getenv("OCR_ENGINE", "tesseract").strip().lower()
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "lao+eng")
OCR_PSM = int(os.getenv("OCR_PSM", "3"))
OCR_TESSDATA_DIR = os.getenv("OCR_TESSDATA_DIR") or None
OCR_MODEL_PATH = os.getenv("OCR_MODEL_PATH") or None
OCR_CALIBRATION_PATH = os.getenv("OCR_CALIBRATION_PATH") or None
OCR_DEVICE = os.getenv("OCR_DEVICE", "cpu").strip().lower()
OCR_DECODER = os.getenv("OCR_DECODER", "greedy").strip().lower()
OCR_BEAM_WIDTH = int(os.getenv("OCR_BEAM_WIDTH", "10"))
OCR_LANGUAGE_MODEL_PATH = os.getenv("OCR_LANGUAGE_MODEL_PATH") or None
OCR_LANGUAGE_MODEL_WEIGHT = float(os.getenv("OCR_LANGUAGE_MODEL_WEIGHT", "0"))
OCR_LANGUAGE_MODEL_TOKEN_BONUS = float(
    os.getenv("OCR_LANGUAGE_MODEL_TOKEN_BONUS", "0")
)
OCR_LAYOUT_DETECTOR = os.getenv(
    "OCR_LAYOUT_DETECTOR",
    "morphology",
).strip().lower()
OCR_LAYOUT_MODEL_PATH = os.getenv("OCR_LAYOUT_MODEL_PATH") or None
OCR_LAYOUT_CONFIDENCE = float(
    os.getenv("OCR_LAYOUT_CONFIDENCE", "0.55")
)
OCR_READING_ORDER = os.getenv(
    "OCR_READING_ORDER",
    "deterministic",
).strip().lower()
OCR_READING_ORDER_MODEL_PATH = os.getenv(
    "OCR_READING_ORDER_MODEL_PATH"
) or None
OCR_READING_ORDER_MAX_BLOCKS = int(
    os.getenv("OCR_READING_ORDER_MAX_BLOCKS", "256")
)
JOB_ROOT = Path(
    os.getenv(
        "JOB_ROOT",
        str(Path(tempfile.gettempdir()) / "lao-document-ocr-jobs"),
    )
)
RESULT_STORAGE_BACKEND = os.getenv(
    "RESULT_STORAGE_BACKEND",
    "filesystem",
).strip().lower()
RESULT_STORAGE_ROOT = Path(
    os.getenv(
        "RESULT_STORAGE_ROOT",
        str(Path(tempfile.gettempdir()) / "lao-document-ocr-results"),
    )
)
RESULT_STORAGE_S3_BUCKET = os.getenv("RESULT_STORAGE_S3_BUCKET", "").strip()
RESULT_STORAGE_S3_PREFIX = os.getenv("RESULT_STORAGE_S3_PREFIX", "").strip()
RESULT_STORAGE_S3_ENDPOINT_URL = os.getenv("RESULT_STORAGE_S3_ENDPOINT_URL")
RESULT_STORAGE_S3_REGION = os.getenv("RESULT_STORAGE_S3_REGION")
RESULT_STORAGE_S3_FORCE_PATH_STYLE = os.getenv(
    "RESULT_STORAGE_S3_FORCE_PATH_STYLE",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}
JOB_MAX_WORKERS = int(os.getenv("JOB_MAX_WORKERS", "2"))
JOB_MAX_ACTIVE = int(os.getenv("JOB_MAX_ACTIVE", "8"))
JOB_RETENTION_SECONDS = int(os.getenv("JOB_RETENTION_SECONDS", "3600"))
BATCH_MAX_FILES = int(os.getenv("BATCH_MAX_FILES", "10"))
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "0"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_MAX_CLIENTS = int(os.getenv("RATE_LIMIT_MAX_CLIENTS", "10000"))
RATE_LIMIT_TRUST_PROXY_HEADERS = os.getenv(
    "RATE_LIMIT_TRUST_PROXY_HEADERS",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}
ALLOWED_ORIGINS = [
    item.strip()
    for item in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    if item.strip()
]


app = FastAPI(
    title="Lao Document OCR",
    version="0.1.0",
    description="Open-source Lao-first document OCR and editable DOCX export.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

API_METRICS = ApiMetrics()
SUBMISSION_RATE_LIMITER = SlidingWindowRateLimiter(
    requests=RATE_LIMIT_REQUESTS,
    window_seconds=RATE_LIMIT_WINDOW_SECONDS,
    max_clients=RATE_LIMIT_MAX_CLIENTS,
)


def _content_security_policy(path: str) -> str:
    if path in {"/docs", "/redoc"}:
        return (
            "default-src 'none'; "
            "script-src 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; "
            "img-src data: https://fastapi.tiangolo.com; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'none'"
        )
    return "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"


def _request_id(request: Request) -> str:
    supplied = request.headers.get("x-request-id", "").strip()
    if supplied and len(supplied) <= 128 and all(
        char.isalnum() or char in "._-" for char in supplied
    ):
        return supplied
    return uuid.uuid4().hex


@app.middleware("http")
async def observe_request(request: Request, call_next):
    request_id = _request_id(request)
    timer = RequestTimer()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        API_METRICS.observe_request(
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            duration_seconds=timer.elapsed(),
        )
        raise

    API_METRICS.observe_request(
        method=request.method,
        path=request.url.path,
        status_code=status_code,
        duration_seconds=timer.elapsed(),
    )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=()"
    )
    response.headers["Content-Security-Policy"] = _content_security_policy(
        request.url.path
    )
    response.headers["Cache-Control"] = "no-store"
    return response


def _build_result_storage():
    if RESULT_STORAGE_BACKEND == "filesystem":
        return FilesystemArtifactStorage(RESULT_STORAGE_ROOT)
    if RESULT_STORAGE_BACKEND == "s3":
        if not RESULT_STORAGE_S3_BUCKET:
            raise RuntimeError(
                "RESULT_STORAGE_S3_BUCKET is required when RESULT_STORAGE_BACKEND=s3"
            )
        return S3ArtifactStorage(
            RESULT_STORAGE_S3_BUCKET,
            prefix=RESULT_STORAGE_S3_PREFIX,
            endpoint_url=RESULT_STORAGE_S3_ENDPOINT_URL,
            region_name=RESULT_STORAGE_S3_REGION,
            force_path_style=RESULT_STORAGE_S3_FORCE_PATH_STYLE,
        )
    raise RuntimeError(
        f"Unsupported RESULT_STORAGE_BACKEND: {RESULT_STORAGE_BACKEND}"
    )


RESULT_STORAGE = _build_result_storage()


def _download_headers(filename: str) -> dict[str, str]:
    name = Path(filename).name or "result.zip"
    fallback = "".join(
        char
        if char.isascii() and (char.isalnum() or char in "._-")
        else "_"
        for char in name
    ) or "result.zip"
    encoded = quote(name, safe="")
    return {
        "Content-Disposition": (
            f'attachment; filename="{fallback}"; filename*=UTF-8\'\'{encoded}'
        )
    }


def _client_rate_limit_key(request: Request) -> str:
    if RATE_LIMIT_TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            candidate = forwarded.split(",", 1)[0].strip()
            if candidate:
                return candidate[:128]
    if request.client is not None and request.client.host:
        return request.client.host[:128]
    return "unknown"


def _enforce_submission_rate_limit(
    request: Request,
    *,
    cost: int = 1,
) -> None:
    decision = SUBMISSION_RATE_LIMITER.check(
        _client_rate_limit_key(request),
        cost=cost,
    )
    if decision.allowed:
        return
    raise HTTPException(
        status_code=429,
        detail="OCR submission rate limit exceeded.",
        headers={"Retry-After": str(decision.retry_after_seconds)},
    )


@lru_cache(maxsize=8)
def _cached_owned_engine(
    model_path: str,
    calibration_path: str | None,
    device: str,
    decoder: str,
    beam_width: int,
    language_model_path: str | None,
    language_model_weight: float,
    language_model_token_bonus: float,
    region_detector_name: str,
    layout_model_path: str | None,
    layout_confidence_threshold: float,
) -> OwnedRecognizerEngine:
    return OwnedRecognizerEngine(
        model_path,
        calibration_path=calibration_path,
        device=device,
        decoder=decoder,
        beam_width=beam_width,
        language_model_path=language_model_path,
        language_model_weight=language_model_weight,
        language_model_token_bonus=language_model_token_bonus,
        region_detector_name=region_detector_name,
        layout_model_path=layout_model_path,
        layout_confidence_threshold=layout_confidence_threshold,
    )


@lru_cache(maxsize=4)
def _cached_reading_order_resolver(
    model_path: str,
    device: str,
    max_blocks: int,
):
    from lao_document_ocr.reading_order_inference import (
        ExportedReadingOrderResolver,
    )

    return ExportedReadingOrderResolver(
        model_path,
        device=device,
        max_blocks=max_blocks,
    )


def _reading_order_resolver():
    if OCR_READING_ORDER == "deterministic":
        return None
    if OCR_READING_ORDER == "learned":
        if not OCR_READING_ORDER_MODEL_PATH:
            raise OcrEngineError(
                "OCR_READING_ORDER_MODEL_PATH is required when "
                "OCR_READING_ORDER=learned"
            )
        try:
            return _cached_reading_order_resolver(
                OCR_READING_ORDER_MODEL_PATH,
                OCR_DEVICE,
                OCR_READING_ORDER_MAX_BLOCKS,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise OcrEngineError(
                f"Could not load learned reading-order resolver: {exc}"
            ) from exc
    raise OcrEngineError(
        "OCR_READING_ORDER must be 'deterministic' or 'learned'"
    )


def _engine() -> OcrEngine:
    if OCR_ENGINE == "tesseract":
        return TesseractEngine(
            languages=OCR_LANGUAGES,
            psm=OCR_PSM,
            tessdata_dir=OCR_TESSDATA_DIR,
        )
    if OCR_ENGINE == "owned":
        if not OCR_MODEL_PATH:
            raise OcrEngineError("OCR_MODEL_PATH is required when OCR_ENGINE=owned")
        return _cached_owned_engine(
            OCR_MODEL_PATH,
            OCR_CALIBRATION_PATH,
            OCR_DEVICE,
            OCR_DECODER,
            OCR_BEAM_WIDTH,
            OCR_LANGUAGE_MODEL_PATH,
            OCR_LANGUAGE_MODEL_WEIGHT,
            OCR_LANGUAGE_MODEL_TOKEN_BONUS,
            OCR_LAYOUT_DETECTOR,
            OCR_LAYOUT_MODEL_PATH,
            OCR_LAYOUT_CONFIDENCE,
        )
    raise OcrEngineError(f"Unsupported OCR_ENGINE: {OCR_ENGINE}")


def _safe_filename(filename: str | None) -> str:
    return sanitize_filename(filename)


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    written = 0
    with destination.open("wb") as output:
        os.chmod(destination, 0o600)
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
                )
            output.write(chunk)


def _validate_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise HTTPException(status_code=415, detail=f"Unsupported file type. Allowed: {allowed}")
    return suffix


def _validate_saved_upload(path: Path, suffix: str) -> None:
    try:
        validate_uploaded_content(
            path,
            suffix,
            max_pages=MAX_PAGES,
            max_page_pixels=MAX_PAGE_PIXELS,
        )
    except UploadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _write_outputs_archive(
    document,
    work_dir: Path,
    filename: str,
) -> Path:
    stem = Path(filename).stem or "document"
    generated = [
        export_docx(document, work_dir / f"{stem}.docx"),
        export_markdown(document, work_dir / f"{stem}.md"),
        export_text(document, work_dir / f"{stem}.txt"),
        export_json(document, work_dir / f"{stem}.json"),
    ]
    archive_path = work_dir / f"{stem}-ocr.zip"
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as output_zip:
        for path in generated:
            output_zip.write(path, arcname=path.name)
    return archive_path


def _run_conversion_job(record: JobRecord, cancel_event) -> StoredArtifact:
    try:
        document = process_document(
            record.input_path,
            source_name=record.filename,
            engine=_engine(),
            max_pages=MAX_PAGES,
            max_page_pixels=MAX_PAGE_PIXELS,
            should_cancel=cancel_event.is_set,
            reading_order_resolver=_reading_order_resolver(),
            auto_orient_right_angles=record.auto_orient_right_angles,
        )
    except DocumentProcessingCancelled as exc:
        raise JobCancelledError(str(exc)) from exc
    except DocumentProcessingError as exc:
        raise JobPublicError(str(exc)) from exc
    if cancel_event.is_set():
        raise JobCancelledError("Document processing was cancelled.")

    archive = _write_outputs_archive(document, record.workspace, record.filename)
    artifact = RESULT_STORAGE.put_file(
        archive,
        key=f"jobs/{record.id}/{archive.name}",
        filename=archive.name,
        media_type="application/zip",
    )
    if cancel_event.is_set():
        RESULT_STORAGE.delete(artifact)
        raise JobCancelledError("Document processing was cancelled.")
    return artifact


JOB_MANAGER = ConversionJobManager(
    JOB_ROOT,
    _run_conversion_job,
    max_workers=JOB_MAX_WORKERS,
    max_active_jobs=JOB_MAX_ACTIVE,
    retention_seconds=JOB_RETENTION_SECONDS,
    artifact_exists=RESULT_STORAGE.exists,
    artifact_cleanup=RESULT_STORAGE.delete,
)


@app.get("/metrics", include_in_schema=False)
def metrics() -> PlainTextResponse:
    return PlainTextResponse(
        API_METRICS.render_prometheus(JOB_MANAGER.snapshot()),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/health")
def health() -> dict:
    try:
        engine = _engine()
        ready = engine.is_available()
        metadata = engine.metadata()
        resolver = _reading_order_resolver()
        reading_order_metadata = (
            resolver.metadata()
            if resolver is not None
            else DeterministicReadingOrderResolver().metadata()
        )
        error = None
    except Exception as exc:
        engine = None
        ready = False
        metadata = None
        reading_order_metadata = None
        error = str(exc)

    payload = {
        "status": "ok",
        "ocr_ready": ready,
        "engine": OCR_ENGINE,
        "metadata": metadata,
        "reading_order": reading_order_metadata,
        "error": error,
        "limits": {
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "max_pages": MAX_PAGES,
            "max_page_pixels": MAX_PAGE_PIXELS,
        },
        "jobs": {
            "max_workers": JOB_MAX_WORKERS,
            "max_active_jobs": JOB_MAX_ACTIVE,
            "retention_seconds": JOB_RETENTION_SECONDS,
            "batch_max_files": BATCH_MAX_FILES,
        },
        "submission_rate_limit": SUBMISSION_RATE_LIMITER.snapshot(),
        "result_storage": {
            "backend": RESULT_STORAGE.metadata().get("backend", "unknown"),
        },
    }
    if OCR_ENGINE == "tesseract":
        try:
            available_languages = (
                engine.available_languages() if isinstance(engine, TesseractEngine) else []
            )
        except OcrEngineError:
            available_languages = []
        payload.update(
            {
                "required_languages": OCR_LANGUAGES.split("+"),
                "psm": OCR_PSM,
                "available_languages": available_languages,
            }
        )
    return payload


@app.post("/v1/parse")
async def parse_document(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    auto_orient_right_angles: Annotated[bool, Form()] = False,
) -> dict:
    filename = _safe_filename(file.filename)
    _enforce_submission_rate_limit(request)
    suffix = _validate_suffix(filename)

    with tempfile.TemporaryDirectory(prefix="lao-ocr-") as temp_dir:
        input_path = Path(temp_dir) / f"input{suffix}"
        await _save_upload(file, input_path)
        _validate_saved_upload(input_path, suffix)
        try:
            document = process_document(
                input_path,
                source_name=filename,
                engine=_engine(),
                max_pages=MAX_PAGES,
                max_page_pixels=MAX_PAGE_PIXELS,
                reading_order_resolver=_reading_order_resolver(),
                auto_orient_right_angles=auto_orient_right_angles,
            )
        except DocumentProcessingError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return document.model_dump(mode="json", exclude_none=True)


@app.post("/v1/convert")
async def convert_document(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    auto_orient_right_angles: Annotated[bool, Form()] = False,
) -> StreamingResponse:
    filename = _safe_filename(file.filename)
    _enforce_submission_rate_limit(request)
    suffix = _validate_suffix(filename)

    with tempfile.TemporaryDirectory(prefix="lao-ocr-") as temp_dir:
        work_dir = Path(temp_dir)
        input_path = work_dir / f"input{suffix}"
        await _save_upload(file, input_path)
        _validate_saved_upload(input_path, suffix)

        try:
            document = process_document(
                input_path,
                source_name=filename,
                engine=_engine(),
                max_pages=MAX_PAGES,
                max_page_pixels=MAX_PAGE_PIXELS,
                reading_order_resolver=_reading_order_resolver(),
                auto_orient_right_angles=auto_orient_right_angles,
            )
        except DocumentProcessingError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        archive_path = _write_outputs_archive(document, work_dir, filename)
        archive = io.BytesIO(archive_path.read_bytes())
        archive.seek(0)

    headers = _download_headers(f"{Path(filename).stem}-ocr.zip")
    return StreamingResponse(archive, media_type="application/zip", headers=headers)


@app.post("/v1/jobs", status_code=202)
async def create_conversion_job(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    auto_orient_right_angles: Annotated[bool, Form()] = False,
) -> dict:
    filename = _safe_filename(file.filename)
    _enforce_submission_rate_limit(request)
    suffix = _validate_suffix(filename)
    try:
        record = JOB_MANAGER.reserve(
            filename,
            suffix,
            auto_orient_right_angles=auto_orient_right_angles,
        )
    except JobCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    try:
        await _save_upload(file, record.input_path)
        _validate_saved_upload(record.input_path, suffix)
        JOB_MANAGER.enqueue(record.id)
    except Exception:
        JOB_MANAGER.discard(record.id)
        raise
    return JOB_MANAGER.public(record.id)


@app.post("/v1/jobs/batch", status_code=202)
async def create_conversion_batch(
    request: Request,
    files: Annotated[list[UploadFile], File(...)],
    auto_orient_right_angles: Annotated[bool, Form()] = False,
) -> dict:
    if not files:
        raise HTTPException(status_code=422, detail="At least one file is required.")
    if len(files) > BATCH_MAX_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Batch exceeds {BATCH_MAX_FILES} file limit.",
        )

    prepared: list[tuple[UploadFile, str, str]] = []
    for upload in files:
        filename = _safe_filename(upload.filename)
        suffix = _validate_suffix(filename)
        prepared.append((upload, filename, suffix))

    _enforce_submission_rate_limit(request, cost=len(prepared))

    try:
        records = JOB_MANAGER.reserve_many(
            [(filename, suffix) for _, filename, suffix in prepared],
            auto_orient_right_angles=auto_orient_right_angles,
        )
    except JobCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    try:
        for (upload, _, suffix), record in zip(prepared, records, strict=True):
            await _save_upload(upload, record.input_path)
            _validate_saved_upload(record.input_path, suffix)
    except Exception:
        for record in records:
            JOB_MANAGER.discard(record.id)
        raise

    try:
        for record in records:
            JOB_MANAGER.enqueue(record.id)
    except Exception:
        for record in records:
            try:
                JOB_MANAGER.cancel(record.id)
            except JobNotFoundError:
                pass
        raise

    return {
        "count": len(records),
        "jobs": [JOB_MANAGER.public(record.id) for record in records],
    }


@app.get("/v1/jobs/{job_id}")
def get_conversion_job(job_id: str) -> dict:
    try:
        return JOB_MANAGER.public(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc


@app.delete("/v1/jobs/{job_id}")
def cancel_conversion_job(job_id: str) -> dict:
    try:
        return JOB_MANAGER.cancel(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc


@app.get("/v1/jobs/{job_id}/download")
def download_conversion_job(job_id: str):
    try:
        record = JOB_MANAGER.get_record(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc

    if record.status != JobStatus.SUCCEEDED:
        raise HTTPException(
            status_code=409,
            detail=f"Job is not ready for download (status={record.status.value}).",
        )

    if record.output_artifact is not None:
        artifact = record.output_artifact
        if not RESULT_STORAGE.exists(artifact):
            raise HTTPException(
                status_code=410,
                detail="Job result is no longer available.",
            )
        return StreamingResponse(
            RESULT_STORAGE.iter_bytes(artifact),
            media_type=artifact.media_type,
            headers=_download_headers(artifact.filename),
        )

    if record.output_path is None:
        raise HTTPException(
            status_code=409,
            detail="Job completed without a downloadable result.",
        )
    if not record.output_path.is_file():
        raise HTTPException(status_code=410, detail="Job result is no longer available.")

    return FileResponse(
        record.output_path,
        media_type="application/zip",
        filename=record.output_path.name,
    )
