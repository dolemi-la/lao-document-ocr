import json

from lao_document_ocr.layout_benchmark import (
    benchmark_layout,
    load_document_ast,
    write_layout_report,
)
from lao_document_ocr.models import (
    Block,
    BlockType,
    BoundingBox,
    Document,
    Page,
)


def _document() -> Document:
    return Document(
        source_name="fixture.png",
        pages=[
            Page(
                number=1,
                width=600,
                height=800,
                blocks=[
                    Block(
                        type=BlockType.HEADING,
                        text="Heading",
                        bbox=BoundingBox(x=50, y=40, width=500, height=50),
                    ),
                    Block(
                        type=BlockType.PARAGRAPH,
                        text="Body",
                        bbox=BoundingBox(x=60, y=130, width=480, height=100),
                    ),
                ],
            )
        ],
    )


def test_layout_report_round_trip(tmp_path) -> None:
    reference = _document()
    prediction = _document()
    reference_path = tmp_path / "reference.json"
    prediction_path = tmp_path / "prediction.json"
    reference_path.write_text(
        reference.model_dump_json(indent=2),
        encoding="utf-8",
    )
    prediction_path.write_text(
        prediction.model_dump_json(indent=2),
        encoding="utf-8",
    )

    loaded_reference = load_document_ast(reference_path)
    loaded_prediction = load_document_ast(prediction_path)
    report = benchmark_layout(loaded_reference, loaded_prediction)
    output = write_layout_report(report, tmp_path / "layout-report.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1"
    assert payload["iou_threshold"] == 0.5
    assert payload["metrics"]["block_f1"] == 1.0
    assert payload["metrics"]["reading_order_accuracy"] == 1.0
