from __future__ import annotations

from dataclasses import asdict, dataclass

from lao_document_ocr.models import Block, BlockType, Document, Page


@dataclass(frozen=True)
class LayoutBlockMatch:
    reference_index: int
    prediction_index: int
    iou: float


@dataclass(frozen=True)
class LayoutMetrics:
    reference_pages: int
    predicted_pages: int
    reference_blocks: int
    predicted_blocks: int
    matched_blocks: int
    block_precision: float
    block_recall: float
    block_f1: float
    mean_iou: float
    block_type_accuracy: float
    reading_order_accuracy: float
    reference_tables: int
    predicted_tables: int
    matched_tables: int
    table_shape_accuracy: float
    table_cell_precision: float
    table_cell_recall: float
    table_cell_f1: float

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return float(numerator) / float(denominator)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _normalized_box(
    block: Block,
    page: Page,
) -> tuple[float, float, float, float] | None:
    if block.bbox is None or page.width <= 0 or page.height <= 0:
        return None
    box = block.bbox
    return (
        box.x / page.width,
        box.y / page.height,
        (box.x + box.width) / page.width,
        (box.y + box.height) / page.height,
    )


def block_iou(
    reference: Block,
    reference_page: Page,
    prediction: Block,
    prediction_page: Page,
) -> float:
    ref = _normalized_box(reference, reference_page)
    pred = _normalized_box(prediction, prediction_page)
    if ref is None or pred is None:
        return 0.0

    left = max(ref[0], pred[0])
    top = max(ref[1], pred[1])
    right = min(ref[2], pred[2])
    bottom = min(ref[3], pred[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0:
        return 0.0

    reference_area = max(0.0, ref[2] - ref[0]) * max(0.0, ref[3] - ref[1])
    prediction_area = max(0.0, pred[2] - pred[0]) * max(0.0, pred[3] - pred[1])
    union = reference_area + prediction_area - intersection
    return intersection / union if union > 0 else 0.0


def match_page_blocks(
    reference_page: Page,
    prediction_page: Page,
    *,
    iou_threshold: float = 0.5,
) -> list[LayoutBlockMatch]:
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")

    candidates: list[tuple[float, int, int]] = []
    for reference_index, reference in enumerate(reference_page.blocks):
        for prediction_index, prediction in enumerate(prediction_page.blocks):
            iou = block_iou(reference, reference_page, prediction, prediction_page)
            if iou >= iou_threshold:
                candidates.append((iou, reference_index, prediction_index))

    matches: list[LayoutBlockMatch] = []
    used_reference: set[int] = set()
    used_prediction: set[int] = set()

    for iou, reference_index, prediction_index in sorted(candidates, reverse=True):
        if reference_index in used_reference or prediction_index in used_prediction:
            continue
        used_reference.add(reference_index)
        used_prediction.add(prediction_index)
        matches.append(
            LayoutBlockMatch(
                reference_index=reference_index,
                prediction_index=prediction_index,
                iou=iou,
            )
        )

    return sorted(matches, key=lambda match: match.reference_index)


def _table_shape(block: Block) -> tuple[int, int]:
    rows = block.metadata.get("rows")
    columns = block.metadata.get("columns")
    if isinstance(rows, int) and isinstance(columns, int):
        return rows, columns
    if not block.cells:
        return 0, 0
    return (
        max(cell.row + cell.row_span for cell in block.cells),
        max(cell.column + cell.column_span for cell in block.cells),
    )


def _cell_structure(block: Block) -> set[tuple[int, int, int, int]]:
    return {
        (cell.row, cell.column, cell.row_span, cell.column_span)
        for cell in block.cells
    }


def evaluate_layout(
    reference: Document,
    prediction: Document,
    *,
    iou_threshold: float = 0.5,
) -> LayoutMetrics:
    reference_blocks = sum(len(page.blocks) for page in reference.pages)
    predicted_blocks = sum(len(page.blocks) for page in prediction.pages)
    reference_tables = sum(
        1
        for page in reference.pages
        for block in page.blocks
        if block.type == BlockType.TABLE
    )
    predicted_tables = sum(
        1
        for page in prediction.pages
        for block in page.blocks
        if block.type == BlockType.TABLE
    )
    reference_cells = sum(
        len(_cell_structure(block))
        for page in reference.pages
        for block in page.blocks
        if block.type == BlockType.TABLE
    )
    predicted_cells = sum(
        len(_cell_structure(block))
        for page in prediction.pages
        for block in page.blocks
        if block.type == BlockType.TABLE
    )

    matched_blocks = 0
    iou_sum = 0.0
    type_correct = 0
    reading_pair_correct = 0
    reading_pair_total = 0
    matched_tables = 0
    shape_correct = 0
    cell_true_positive = 0

    page_count = min(len(reference.pages), len(prediction.pages))
    for page_index in range(page_count):
        reference_page = reference.pages[page_index]
        prediction_page = prediction.pages[page_index]
        matches = match_page_blocks(
            reference_page,
            prediction_page,
            iou_threshold=iou_threshold,
        )

        matched_blocks += len(matches)
        iou_sum += sum(match.iou for match in matches)

        ordered_matches = sorted(matches, key=lambda match: match.reference_index)
        for left_index, left in enumerate(ordered_matches):
            for right in ordered_matches[left_index + 1 :]:
                reading_pair_total += 1
                if left.prediction_index < right.prediction_index:
                    reading_pair_correct += 1

        for match in matches:
            reference_block = reference_page.blocks[match.reference_index]
            prediction_block = prediction_page.blocks[match.prediction_index]

            if reference_block.type == prediction_block.type:
                type_correct += 1

            if (
                reference_block.type != BlockType.TABLE
                or prediction_block.type != BlockType.TABLE
            ):
                continue

            matched_tables += 1
            if _table_shape(reference_block) == _table_shape(prediction_block):
                shape_correct += 1
            cell_true_positive += len(
                _cell_structure(reference_block) & _cell_structure(prediction_block)
            )

    block_precision = _safe_ratio(matched_blocks, predicted_blocks)
    block_recall = _safe_ratio(matched_blocks, reference_blocks)
    cell_precision = _safe_ratio(cell_true_positive, predicted_cells)
    cell_recall = _safe_ratio(cell_true_positive, reference_cells)

    return LayoutMetrics(
        reference_pages=len(reference.pages),
        predicted_pages=len(prediction.pages),
        reference_blocks=reference_blocks,
        predicted_blocks=predicted_blocks,
        matched_blocks=matched_blocks,
        block_precision=block_precision,
        block_recall=block_recall,
        block_f1=_f1(block_precision, block_recall),
        mean_iou=_safe_ratio(iou_sum, matched_blocks),
        block_type_accuracy=_safe_ratio(type_correct, matched_blocks),
        reading_order_accuracy=_safe_ratio(
            reading_pair_correct,
            reading_pair_total,
        ),
        reference_tables=reference_tables,
        predicted_tables=predicted_tables,
        matched_tables=matched_tables,
        table_shape_accuracy=_safe_ratio(shape_correct, reference_tables),
        table_cell_precision=cell_precision,
        table_cell_recall=cell_recall,
        table_cell_f1=_f1(cell_precision, cell_recall),
    )
