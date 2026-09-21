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
from lao_document_ocr.ocr import TesseractEngine
from lao_document_ocr.pipeline import SUPPORTED_SUFFIXES, DocumentProcessingError, process_document

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_PAGES = int(os.getenv("MAX_PAGES", "60"))
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "lao+eng")
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


def _engine() -> TesseractEngine:
    return TesseractEngine(languages=OCR_LANGUAGES)


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
    engine = _engine()
    try:
        languages = engine.available_languages()
        ready = engine.is_available()
        error = None
    except Exception as exc:
        languages = []
        ready = False
        error = str(exc)
    return {
        "status": "ok",
        "ocr_ready": ready,
        "engine": "tesseract",
        "required_languages": OCR_LANGUAGES.split("+"),
        "available_languages": languages,
        "error": error,
    }


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
