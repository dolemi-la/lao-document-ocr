from pathlib import Path

ROOT = Path(__file__).parents[1]
PYPROJECT = ROOT / "pyproject.toml"
DOCKERFILE = ROOT / "Dockerfile.api"
COMPOSE = ROOT / "docker-compose.yml"


def test_pyproject_has_optional_s3_extra() -> None:
    content = PYPROJECT.read_text(encoding="utf-8")
    assert "s3 = [" in content
    assert '"boto3>=1.35,<2"' in content


def test_api_dockerfile_can_optionally_install_s3_dependencies() -> None:
    content = DOCKERFILE.read_text(encoding="utf-8")
    assert "ARG INSTALL_S3=false" in content
    assert 'if [ "$INSTALL_S3" = "true" ]' in content
    assert 'pip install --timeout 120 --retries 10 ".[s3]"' in content


def test_compose_passes_s3_configuration_and_credentials() -> None:
    content = COMPOSE.read_text(encoding="utf-8")

    required = [
        'INSTALL_S3: "${INSTALL_S3:-false}"',
        'RESULT_STORAGE_S3_BUCKET: "${RESULT_STORAGE_S3_BUCKET:-}"',
        'RESULT_STORAGE_S3_PREFIX: "${RESULT_STORAGE_S3_PREFIX:-}"',
        'RESULT_STORAGE_S3_ENDPOINT_URL: "${RESULT_STORAGE_S3_ENDPOINT_URL:-}"',
        'RESULT_STORAGE_S3_REGION: "${RESULT_STORAGE_S3_REGION:-}"',
        'RESULT_STORAGE_S3_FORCE_PATH_STYLE: "${RESULT_STORAGE_S3_FORCE_PATH_STYLE:-false}"',
        'AWS_ACCESS_KEY_ID: "${AWS_ACCESS_KEY_ID:-}"',
        'AWS_SECRET_ACCESS_KEY: "${AWS_SECRET_ACCESS_KEY:-}"',
        'AWS_SESSION_TOKEN: "${AWS_SESSION_TOKEN:-}"',
    ]
    for expected in required:
        assert expected in content
