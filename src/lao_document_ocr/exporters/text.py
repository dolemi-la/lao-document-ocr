from pathlib import Path

from lao_document_ocr.models import Document


def export_text(document: Document, path: str | Path) -> Path:
    destination = Path(path)
    destination.write_text(document.plain_text + "\n", encoding="utf-8")
    return destination
