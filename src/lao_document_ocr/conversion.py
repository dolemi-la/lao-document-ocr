from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lao_document_ocr.exporters import (
    export_docx,
    export_json,
    export_markdown,
    export_text,
)
from lao_document_ocr.ocr.base import OcrEngine
from lao_document_ocr.pipeline import process_document
from lao_document_ocr.reading_order import ReadingOrderResolver


@dataclass(frozen=True)
class ConversionOutputs:
    docx: Path
    markdown: Path
    text: Path
    json: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "docx": str(self.docx),
            "markdown": str(self.markdown),
            "text": str(self.text),
            "json": str(self.json),
        }


def convert_document_to_outputs(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    engine: OcrEngine,
    max_pages: int = 60,
    font_name: str = "Noto Sans Lao",
    reading_order_resolver: ReadingOrderResolver | None = None,
) -> ConversionOutputs:
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"Input document not found: {source}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = source.stem or "document"

    document = process_document(
        source,
        source_name=source.name,
        engine=engine,
        max_pages=max_pages,
        reading_order_resolver=reading_order_resolver,
    )

    return ConversionOutputs(
        docx=export_docx(document, output / f"{stem}.docx", font_name=font_name),
        markdown=export_markdown(document, output / f"{stem}.md"),
        text=export_text(document, output / f"{stem}.txt"),
        json=export_json(document, output / f"{stem}.json"),
    )
