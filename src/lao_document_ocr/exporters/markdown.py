from pathlib import Path

from lao_document_ocr.models import BlockType, Document


def document_to_markdown(document: Document) -> str:
    output: list[str] = []
    for page in document.pages:
        if len(document.pages) > 1:
            output.append(f"<!-- Page {page.number} -->")
        for block in page.blocks:
            if block.type == BlockType.HEADING:
                level = max(1, min(6, block.level or 1))
                output.append(f"{'#' * level} {block.text.replace(chr(10), ' ')}")
            elif block.type == BlockType.LIST:
                for line in block.text.splitlines():
                    output.append(f"- {line}")
            elif block.type == BlockType.TABLE and block.cells:
                # Full table reconstruction is a later milestone. Keep cell text lossless for now.
                output.append(block.text)
            elif block.text:
                output.append(block.text)
        output.append("")
    return "\n\n".join(part for part in output if part is not None).strip() + "\n"


def export_markdown(document: Document, path: str | Path) -> Path:
    destination = Path(path)
    destination.write_text(document_to_markdown(document), encoding="utf-8")
    return destination
