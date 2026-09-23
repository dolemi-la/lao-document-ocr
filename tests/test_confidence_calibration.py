import json

import pytest

from lao_document_ocr.confidence_calibration import (
    ConfidenceCalibration,
    fit_confidence_calibration,
    fit_from_recognizer_report,
)


def test_fit_calibration_maps_bins_to_observed_accuracy() -> None:
    calibration = fit_confidence_calibration(
        [
            (0.1, 0.0),
            (0.2, 0.1),
            (0.8, 0.9),
            (0.9, 1.0),
        ],
        max_bins=2,
    )

    assert calibration.sample_count == 4
    assert calibration.calibrate(0.15) == pytest.approx(0.05)
    assert calibration.calibrate(0.85) == pytest.approx(0.95)
    assert calibration.calibrated_mae <= calibration.raw_mae


def test_calibration_round_trip(tmp_path) -> None:
    calibration = fit_confidence_calibration([(0.2, 0.1), (0.8, 0.9)], max_bins=2)
    path = calibration.save(tmp_path / "calibration.json")
    loaded = ConfidenceCalibration.load(path)
    assert loaded == calibration


def test_fit_from_recognizer_report_uses_character_accuracy() -> None:
    report = {
        "samples": [
            {"uncalibrated_confidence": 0.3, "cer": 0.8},
            {"uncalibrated_confidence": 0.9, "cer": 0.1},
        ]
    }
    calibration = fit_from_recognizer_report(report, max_bins=2)
    assert calibration.calibrate(0.3) == pytest.approx(0.2)
    assert calibration.calibrate(0.9) == pytest.approx(0.9)


def test_report_requires_enough_samples() -> None:
    with pytest.raises(ValueError, match="fewer than two"):
        fit_from_recognizer_report(
            {"samples": [{"uncalibrated_confidence": 0.5, "cer": 0.5}]}
        )


def test_load_rejects_unknown_schema(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": "99"}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        ConfidenceCalibration.load(path)


def test_fit_from_beam_report_records_decoder() -> None:
    report = {
        "model": {"decoder": "beam"},
        "samples": [
            {"uncalibrated_confidence": 0.3, "cer": 0.8},
            {"uncalibrated_confidence": 0.9, "cer": 0.1},
        ],
    }
    calibration = fit_from_recognizer_report(report, max_bins=2)
    assert calibration.decoder == "beam"


def test_legacy_calibration_without_decoder_loads_as_greedy(tmp_path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "method": "legacy",
                "sample_count": 0,
                "raw_mae": 0.0,
                "calibrated_mae": 0.0,
                "bins": [],
            }
        ),
        encoding="utf-8",
    )
    assert ConfidenceCalibration.load(path).decoder == "greedy"
