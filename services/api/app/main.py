from __future__ import annotations

import io
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

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
from services.api.app.jobs import (
    ConversionJobManager,
    JobCancelledError,
    JobCapacityError,
    JobNotFoundError,
    JobRecord,
    JobStatus,
)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_PAGES = int(os.getenv("MAX_PAGES", "60"))
OCR_ENGINE = os.getenv("OCR_ENGINE", "tesseract").strip().lower()
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "lao+eng")
OCR_PSM = int(os.getenv("OCR_PSM", "3"))
OCR_MODEL_PATH = os.getenv("OCR_MODEL_PATH")
OCR_CALIBRATION_PATH = os.getenv("OCR_CALIBRATION_PATH")
JOB_ROOT = Path(
    os.getenv(
        "JOB_ROOT",
        str(Path(tempfile.gettempdir()) / "lao-document-ocr-jobs"),
    )
)
JOB_MAX_WORKERS = int(os.getenv("JOB_MAX_WORKERS", "2"))
JOB_MAX_ACTIVE = int(os.getenv("JOB_MAX_ACTIVE", "8"))
JOB_RETENTION_SECONDS = int(os.getenv("JOB_RETENTION_SECONDS", "3600"))
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


def _engine() -> OcrEngine:
    if OCR_ENGINE == "tesseract":
        return TesseractEngine(languages=OCR_LANGUAGES, psm=OCR_PSM)
    if OCR_ENGINE == "owned":
        if not OCR_MODEL_PATH:
            raise OcrEngineError("OCR_MODEL_PATH is required when OCR_ENGINE=owned")
        return OwnedRecognizerEngine(
            OCR_MODEL_PATH,
            calibration_path=OCR_CALIBRATION_PATH,
        )
    raise OcrEngineError(f"Unsupported OCR_ENGINE: {OCR_ENGINE}")


def _safe_filename(filename: str | None) -> str:
    name = Path(filename or "document").name
    return name or "document"


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    written = 0
    with destination.open("wb") as output:
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


def _run_conversion_job(record: JobRecord, cancel_event) -> Path:
    try:
        document = process_document(
            record.input_path,
            source_name=record.filename,
            engine=_engine(),
            max_pages=MAX_PAGES,
            should_cancel=cancel_event.is_set,
        )
    except DocumentProcessingCancelled as exc:
        raise JobCancelledError(str(exc)) from exc
    if cancel_event.is_set():
        raise JobCancelledError("Document processing was cancelled.")
    return _write_outputs_archive(document, record.workspace, record.filename)


JOB_MANAGER = ConversionJobManager(
    JOB_ROOT,
    _run_conversion_job,
    max_workers=JOB_MAX_WORKERS,
    max_active_jobs=JOB_MAX_ACTIVE,
    retention_seconds=JOB_RETENTION_SECONDS,
)


@app.get("/health")
def health() -> dict:
    try:
        engine = _engine()
        ready = engine.is_available()
        metadata = engine.metadata()
        error = None
    except Exception as exc:
        engine = None
        ready = False
        metadata = None
        error = str(exc)

    payload = {
        "status": "ok",
        "ocr_ready": ready,
        "engine": OCR_ENGINE,
        "metadata": metadata,
        "error": error,
        "jobs": {
            "max_workers": JOB_MAX_WORKERS,
            "max_active_jobs": JOB_MAX_ACTIVE,
            "retention_seconds": JOB_RETENTION_SECONDS,
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
async def parse_document(file: Annotated[UploadFile, File(...)]) -> dict:
    filename = _safe_filename(file.filename)
    suffix = _validate_suffix(filename)

    with tempfile.TemporaryDirectory(prefix="lao-ocr-") as temp_dir:
        input_path = Path(temp_dir) / f"input{suffix}"
        await _save_upload(file, input_path)
        try:
            document = process_document(
                input_path,
                source_name=filename,
                engine=_engine(),
                max_pages=MAX_PAGES,
            )
        except DocumentProcessingError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return document.model_dump(mode="json", exclude_none=True)


@app.post("/v1/convert")
async def convert_document(file: Annotated[UploadFile, File(...)]) -> StreamingResponse:
    filename = _safe_filename(file.filename)
    suffix = _validate_suffix(filename)

    with tempfile.TemporaryDirectory(prefix="lao-ocr-") as temp_dir:
        work_dir = Path(temp_dir)
        input_path = work_dir / f"input{suffix}"
        await _save_upload(file, input_path)

        try:
            document = process_document(
                input_path,
                source_name=filename,
                engine=_engine(),
                max_pages=MAX_PAGES,
            )
        except DocumentProcessingError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        archive_path = _write_outputs_archive(document, work_dir, filename)
        archive = io.BytesIO(archive_path.read_bytes())
        archive.seek(0)

    headers = {"Content-Disposition": f'attachment; filename="{Path(filename).stem}-ocr.zip"'}
    return StreamingResponse(archive, media_type="application/zip", headers=headers)


@app.post("/v1/jobs", status_code=202)
async def create_conversion_job(file: Annotated[UploadFile, File(...)]) -> dict:
    filename = _safe_filename(file.filename)
    suffix = _validate_suffix(filename)
    try:
        record = JOB_MANAGER.reserve(filename, suffix)
    except JobCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    try:
        await _save_upload(file, record.input_path)
        JOB_MANAGER.enqueue(record.id)
    except Exception:
        JOB_MANAGER.discard(record.id)
        raise
    return JOB_MANAGER.public(record.id)


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

    if record.status != JobStatus.SUCCEEDED or record.output_path is None:
        raise HTTPException(
            status_code=409,
            detail=f"Job is not ready for download (status={record.status.value}).",
        )
    if not record.output_path.is_file():
        raise HTTPException(status_code=410, detail="Job result is no longer available.")

    return FileResponse(
        record.output_path,
        media_type="application/zip",
        filename=record.output_path.name,
    )
