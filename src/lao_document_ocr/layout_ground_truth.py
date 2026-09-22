from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from lao_document_ocr.models import BlockType, Document


class LayoutGroundTruthError(ValueError):
    pass


def load_layout_ground_truth(path: str | Path) -> Document:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LayoutGroundTruthError(
            f"Could not read layout ground truth: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise LayoutGroundTruthError(
            f"Invalid layout ground truth JSON: {exc}"
        ) from exc

    try:
        return Document.model_validate(payload)
    except ValidationError as exc:
        raise LayoutGroundTruthError(
            f"Invalid layout ground truth schema: {exc}"
        ) from exc


def validate_layout_ground_truth(
    document: Document,
    *,
    image_size: tuple[int, int] | None = None,
) -> list[str]:
    errors: list[str] = []

    if len(document.pages) != 1:
        errors.append(
            "layout ground truth must contain exactly one page"
        )
        return errors

    page = document.pages[0]
    if page.width < 1 or page.height < 1:
        errors.append("layout page dimensions must be positive")

    if image_size is not None:
        width, height = image_size
        if (page.width, page.height) != (width, height):
            errors.append(
                "layout page dimensions do not match source image "
                f"({page.width}x{page.height} != {width}x{height})"
            )

    for block_index, block in enumerate(page.blocks):
        if block.bbox is None:
            errors.append(
                f"block {block_index}: bbox is required for layout ground truth"
            )
            continue

        box = block.bbox
        if box.x < 0 or box.y < 0:
            errors.append(
                f"block {block_index}: bbox coordinates must be non-negative"
            )
        if box.width < 1 or box.height < 1:
            errors.append(
                f"block {block_index}: bbox dimensions must be positive"
            )
        if (
            box.x + box.width > page.width
            or box.y + box.height > page.height
        ):
            errors.append(
                f"block {block_index}: bbox exceeds page bounds"
            )

        if block.type != BlockType.TABLE:
            continue

        rows = block.metadata.get("rows")
        columns = block.metadata.get("columns")
        if not isinstance(rows, int) or rows < 1:
            errors.append(
                f"block {block_index}: table metadata.rows must be a positive integer"
            )
            rows = None
        if not isinstance(columns, int) or columns < 1:
            errors.append(
                f"block {block_index}: table metadata.columns must be a positive integer"
            )
            columns = None

        occupied: set[tuple[int, int]] = set()
        for cell_index, cell in enumerate(block.cells):
            if cell.row < 0 or cell.column < 0:
                errors.append(
                    f"block {block_index} cell {cell_index}: row/column must be non-negative"
                )
                continue
            if cell.row_span < 1 or cell.column_span < 1:
                errors.append(
                    f"block {block_index} cell {cell_index}: spans must be positive"
                )
                continue

            if rows is not None and cell.row + cell.row_span > rows:
                errors.append(
                    f"block {block_index} cell {cell_index}: row span exceeds table rows"
                )
            if (
                columns is not None
                and cell.column + cell.column_span > columns
            ):
                errors.append(
                    f"block {block_index} cell {cell_index}: column span exceeds table columns"
                )

            for row in range(cell.row, cell.row + cell.row_span):
                for column in range(
                    cell.column,
                    cell.column + cell.column_span,
                ):
                    position = (row, column)
                    if position in occupied:
                        errors.append(
                            f"block {block_index} cell {cell_index}: overlaps another cell"
                        )
                    occupied.add(position)

    return errors


def write_layout_ground_truth(
    document: Document,
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        document.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
