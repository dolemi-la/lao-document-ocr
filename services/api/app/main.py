from __future__ import annotations

import io
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

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
from lao_document_ocr.pipeline import SUPPORTED_SUFFIXES, DocumentProcessingError, process_document

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_PAGES = int(os.getenv("MAX_PAGES", "60"))
OCR_ENGINE = os.getenv("OCR_ENGINE", "tesseract").strip().lower()
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "lao+eng")
OCR_PSM = int(os.getenv("OCR_PSM", "3"))
OCR_MODEL_PATH = os.getenv("OCR_MODEL_PATH")
OCR_CALIBRATION_PATH = os.getenv("OCR_CALIBRATION_PATH")
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
    allow_methods=["GET", "POST"],
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

        stem = Path(filename).stem or "document"
        generated = [
            export_docx(document, work_dir / f"{stem}.docx"),
            export_markdown(document, work_dir / f"{stem}.md"),
            export_text(document, work_dir / f"{stem}.txt"),
            export_json(document, work_dir / f"{stem}.json"),
        ]

        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output_zip:
            for path in generated:
                output_zip.write(path, arcname=path.name)
        archive.seek(0)

    headers = {"Content-Disposition": f'attachment; filename="{Path(filename).stem}-ocr.zip"'}
    return StreamingResponse(archive, media_type="application/zip", headers=headers)
