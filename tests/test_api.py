from fastapi.testclient import TestClient

from services.api.app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["engine"] == "tesseract"
    assert payload["psm"] == 3


def test_rejects_unsupported_upload() -> None:
    response = client.post(
        "/v1/parse",
        files={"file": ("document.exe", b"not a document", "application/octet-stream")},
    )
    assert response.status_code == 415


def test_owned_engine_health_requires_model(monkeypatch) -> None:
    import services.api.app.main as api_main

    monkeypatch.setattr(api_main, "OCR_ENGINE", "owned")
    monkeypatch.setattr(api_main, "OCR_MODEL_PATH", None)

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["engine"] == "owned"
    assert payload["ocr_ready"] is False
    assert "OCR_MODEL_PATH" in payload["error"]
