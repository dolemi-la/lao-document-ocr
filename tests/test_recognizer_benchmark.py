from dataclasses import dataclass

from PIL import Image

from lao_document_ocr.recognizer_benchmark import benchmark_recognizer
from lao_document_ocr.training_manifest import TrainingSample


@dataclass
class FakeResult:
    text: str
    confidence: float = 0.8


class FakeRecognizer:
    def __init__(self, outputs: dict[str, str]) -> None:
        self.outputs = outputs

    def recognize(self, image_path):
        return FakeResult(self.outputs[image_path.name])


def test_recognizer_benchmark_aggregates_cer_and_wer(tmp_path) -> None:
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    Image.new("L", (10, 10), 255).save(first)
    Image.new("L", (10, 10), 255).save(second)

    samples = [
        TrainingSample(id="a", image=first, text="abc"),
        TrainingSample(id="b", image=second, text="one two"),
    ]
    recognizer = FakeRecognizer({"a.png": "abc", "b.png": "one too"})

    report = benchmark_recognizer(samples, recognizer)

    assert report["overall"]["samples"] == 2
    assert report["overall"]["character_edits"] == 1
    assert report["overall"]["word_edits"] == 1
    assert report["samples"][0]["uncalibrated_confidence"] == 0.8
