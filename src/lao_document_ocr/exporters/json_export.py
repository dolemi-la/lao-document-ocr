from pathlib import Path

from lao_document_ocr.models import Document


def export_json(document: Document, path: str | Path) -> Path:
    destination = Path(path)
    destination.write_text(
        document.model_dump_json(indent=2, exclude_none=True),
        encoding="utf-8",
    )
    return destination
