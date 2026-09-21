from PIL import Image

from lao_document_ocr.benchmarking import MetricCounts, benchmark_dataset
from lao_document_ocr.dataset import DatasetSample, DatasetSplit, DatasetSubset
from lao_document_ocr.models import BoundingBox
from lao_document_ocr.ocr.base import OcrEngine, RecognizedLine


class FixedEngine(OcrEngine):
    def is_available(self) -> bool:
        return True

    def recognize(self, image: Image.Image) -> list[RecognizedLine]:
        return [
            RecognizedLine(
                text="hello world",
                bbox=BoundingBox(x=10, y=10, width=100, height=20),
                confidence=1.0,
                block_id=1,
                paragraph_id=1,
                line_id=1,
            )
        ]


def test_metric_counts_weight_edits_by_reference_size() -> None:
    counts = MetricCounts()
    counts.add("abc", "axc")
    counts.add("abcdefghij", "abcdefghij")

    assert counts.character_edits == 1
    assert counts.characters == 13
    assert counts.cer == 1 / 13


def test_benchmark_dataset_reports_subset_metrics(tmp_path) -> None:
    image_path = tmp_path / "page.png"
    Image.new("RGB", (300, 120), "white").save(image_path)
    truth_path = tmp_path / "page.txt"
    truth_path.write_text("hello world", encoding="utf-8")

    sample = DatasetSample(
        id="sample-001",
        document_id="document-001",
        split=DatasetSplit.TEST,
        subset=DatasetSubset.CLEAN_PRINT,
        source="page.png",
        ground_truth="page.txt",
        license="CC0-1.0",
        provenance="Synthetic unit test",
        tags=["capture:phone-photo", "language:mixed"],
    )

    report = benchmark_dataset([sample], tmp_path, FixedEngine())

    assert report["overall"]["cer"] == 0
    assert report["overall"]["wer"] == 0
    assert report["subsets"]["clean-print"]["samples"] == 1
    assert report["tags"]["capture:phone-photo"]["samples"] == 1
    assert report["tags"]["language:mixed"]["cer"] == 0
    assert report["samples"][0]["tags"] == [
        "capture:phone-photo",
        "language:mixed",
    ]
    assert report["samples"][0]["id"] == "sample-001"
