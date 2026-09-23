from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CalibrationBin:
    min_confidence: float
    max_confidence: float
    mean_confidence: float
    mean_accuracy: float
    count: int


@dataclass(frozen=True)
class ConfidenceCalibration:
    schema_version: str
    method: str
    sample_count: int
    bins: tuple[CalibrationBin, ...]
    raw_mae: float
    calibrated_mae: float
    decoder: str = "greedy"

    def calibrate(self, confidence: float) -> float:
        if not self.bins:
            return max(0.0, min(1.0, confidence))

        confidence = max(0.0, min(1.0, confidence))
        for bucket in self.bins:
            if bucket.min_confidence <= confidence <= bucket.max_confidence:
                return bucket.mean_accuracy

        nearest = min(
            self.bins,
            key=lambda bucket: abs(bucket.mean_confidence - confidence),
        )
        return nearest.mean_accuracy

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "sample_count": self.sample_count,
            "raw_mae": self.raw_mae,
            "calibrated_mae": self.calibrated_mae,
            "decoder": self.decoder,
            "bins": [asdict(bucket) for bucket in self.bins],
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> ConfidenceCalibration:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != "1":
            raise ValueError("Unsupported confidence calibration schema")
        bins = tuple(CalibrationBin(**item) for item in payload.get("bins", []))
        return cls(
            schema_version="1",
            method=str(payload.get("method", "")),
            sample_count=int(payload.get("sample_count", 0)),
            bins=bins,
            raw_mae=float(payload.get("raw_mae", 0.0)),
            calibrated_mae=float(payload.get("calibrated_mae", 0.0)),
            decoder=str(payload.get("decoder", "greedy")),
        )


def _validate_point(confidence: float, accuracy: float) -> tuple[float, float]:
    if not math.isfinite(confidence) or not math.isfinite(accuracy):
        raise ValueError("Calibration points must be finite")
    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1")
    if not 0 <= accuracy <= 1:
        raise ValueError("Accuracy must be between 0 and 1")
    return confidence, accuracy


def fit_confidence_calibration(
    points: list[tuple[float, float]],
    *,
    max_bins: int = 10,
    decoder: str = "greedy",
) -> ConfidenceCalibration:
    if len(points) < 2:
        raise ValueError("At least two calibration points are required")
    if max_bins < 1:
        raise ValueError("max_bins must be at least 1")
    decoder = decoder.strip().lower()
    if decoder not in {"greedy", "beam"}:
        raise ValueError("decoder must be one of: greedy, beam")

    validated = sorted(_validate_point(*point) for point in points)
    bin_count = min(max_bins, len(validated))
    chunk_size = math.ceil(len(validated) / bin_count)

    bins: list[CalibrationBin] = []
    for start in range(0, len(validated), chunk_size):
        chunk = validated[start : start + chunk_size]
        confidences = [item[0] for item in chunk]
        accuracies = [item[1] for item in chunk]
        bins.append(
            CalibrationBin(
                min_confidence=min(confidences),
                max_confidence=max(confidences),
                mean_confidence=sum(confidences) / len(confidences),
                mean_accuracy=sum(accuracies) / len(accuracies),
                count=len(chunk),
            )
        )

    provisional = ConfidenceCalibration(
        schema_version="1",
        method="quantile-bin-character-accuracy",
        sample_count=len(validated),
        bins=tuple(bins),
        raw_mae=0.0,
        calibrated_mae=0.0,
        decoder=decoder,
    )
    raw_mae = sum(abs(confidence - accuracy) for confidence, accuracy in validated) / len(
        validated
    )
    calibrated_mae = sum(
        abs(provisional.calibrate(confidence) - accuracy)
        for confidence, accuracy in validated
    ) / len(validated)

    return ConfidenceCalibration(
        schema_version="1",
        method=provisional.method,
        sample_count=len(validated),
        bins=tuple(bins),
        raw_mae=raw_mae,
        calibrated_mae=calibrated_mae,
        decoder=decoder,
    )


def fit_from_recognizer_report(
    report: dict,
    *,
    max_bins: int = 10,
) -> ConfidenceCalibration:
    points: list[tuple[float, float]] = []
    for sample in report.get("samples", []):
        confidence = sample.get("uncalibrated_confidence")
        cer = sample.get("cer")
        if confidence is None or cer is None:
            continue
        accuracy = max(0.0, min(1.0, 1.0 - float(cer)))
        points.append((float(confidence), accuracy))
    if len(points) < 2:
        raise ValueError("Recognizer report has fewer than two usable calibration samples")
    model = report.get("model")
    decoder = (
        str(model.get("decoder", "greedy"))
        if isinstance(model, dict)
        else "greedy"
    )
    return fit_confidence_calibration(
        points,
        max_bins=max_bins,
        decoder=decoder,
    )
