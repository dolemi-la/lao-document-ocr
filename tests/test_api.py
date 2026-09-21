from fastapi.testclient import TestClient

from services.api.app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["engine"] == "tesseract"


def test_rejects_unsupported_upload() -> None:
    response = client.post(
        "/v1/parse",
        files={"file": ("document.exe", b"not a document", "application/octet-stream")},
    )
    assert response.status_code == 415
