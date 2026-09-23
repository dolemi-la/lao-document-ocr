import hashlib

from PIL import Image

import lao_document_ocr.ocr.tesseract as tesseract_module
from lao_document_ocr.ocr.tesseract import TesseractEngine


def _traineddata_dir(tmp_path):
    directory = tmp_path / "custom tessdata"
    directory.mkdir()
    (directory / "lao.traineddata").write_bytes(b"lao-model")
    (directory / "eng.traineddata").write_bytes(b"eng-model")
    return directory


def test_custom_tessdata_dir_is_used_for_language_discovery(
    tmp_path,
    monkeypatch,
) -> None:
    directory = _traineddata_dir(tmp_path)
    observed = {}

    def fake_get_languages(config=""):
        observed["config"] = config
        return ["eng", "lao"]

    monkeypatch.setattr(
        tesseract_module.pytesseract,
        "get_languages",
        fake_get_languages,
    )

    engine = TesseractEngine(
        languages="lao+eng",
        tessdata_dir=directory,
    )

    assert engine.is_available() is True
    assert "--tessdata-dir" in observed["config"]
    assert str(directory) in observed["config"]


def test_metadata_hashes_custom_traineddata(tmp_path, monkeypatch) -> None:
    directory = _traineddata_dir(tmp_path)
    monkeypatch.setattr(
        tesseract_module.pytesseract,
        "get_tesseract_version",
        lambda: "5.5.0",
    )

    engine = TesseractEngine(
        languages="lao+eng",
        psm=3,
        tessdata_dir=directory,
    )
    metadata = engine.metadata()

    assert metadata["tessdata"]["source"] == "custom"
    assert metadata["tessdata"]["traineddata_sha256"] == {
        "eng": hashlib.sha256(b"eng-model").hexdigest(),
        "lao": hashlib.sha256(b"lao-model").hexdigest(),
    }


def test_recognize_passes_psm_and_tessdata_config(
    tmp_path,
    monkeypatch,
) -> None:
    directory = _traineddata_dir(tmp_path)
    observed = {}

    monkeypatch.setattr(
        tesseract_module.pytesseract,
        "get_languages",
        lambda config="": ["eng", "lao"],
    )

    def fake_image_to_data(
        image,
        *,
        lang,
        config,
        output_type,
    ):
        observed.update(
            {
                "lang": lang,
                "config": config,
                "output_type": output_type,
            }
        )
        return {
            "text": ["ສະບາຍດີ"],
            "block_num": [1],
            "par_num": [1],
            "line_num": [1],
            "conf": ["95"],
            "left": [10],
            "top": [12],
            "width": [80],
            "height": [20],
        }

    monkeypatch.setattr(
        tesseract_module.pytesseract,
        "image_to_data",
        fake_image_to_data,
    )

    engine = TesseractEngine(
        languages="lao+eng",
        psm=6,
        tessdata_dir=directory,
    )
    lines = engine.recognize(Image.new("RGB", (120, 60), "white"))

    assert observed["lang"] == "lao+eng"
    assert "--psm 6" in observed["config"]
    assert "--tessdata-dir" in observed["config"]
    assert str(directory) in observed["config"]
    assert lines[0].text == "ສະບາຍດີ"
    assert lines[0].confidence == 0.95


def test_invalid_custom_tessdata_dir_is_rejected(tmp_path) -> None:
    missing = tmp_path / "missing"
    try:
        TesseractEngine(tessdata_dir=missing)
    except ValueError as exc:
        assert "tessdata directory not found" in str(exc)
    else:
        raise AssertionError("expected invalid tessdata directory to be rejected")
