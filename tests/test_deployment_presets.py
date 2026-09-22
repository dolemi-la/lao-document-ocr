from pathlib import Path

ROOT = Path(__file__).parents[1]
COMPOSE = ROOT / "docker-compose.yml"
PUBLIC = ROOT / "deploy" / "compose.public.yml"
S3 = ROOT / "deploy" / "compose.s3.yml"
LOCAL_ENV = ROOT / "deploy" / "presets" / "local.env.example"
PUBLIC_ENV = ROOT / "deploy" / "presets" / "public-single-node.env.example"
S3_ENV = ROOT / "deploy" / "presets" / "s3.env.example"


def test_base_compose_supports_configurable_bindings_and_api_url() -> None:
    content = COMPOSE.read_text(encoding="utf-8")
    assert '"${API_BIND_ADDRESS:-0.0.0.0}:${API_PORT:-8000}:8000"' in content
    assert '"${WEB_BIND_ADDRESS:-0.0.0.0}:${WEB_PORT:-5173}:80"' in content
    assert 'VITE_API_URL: "${VITE_API_URL:-http://localhost:8000}"' in content


def test_public_override_hardens_api_container() -> None:
    content = PUBLIC.read_text(encoding="utf-8")
    assert "read_only: true" in content
    assert "no-new-privileges:true" in content
    assert "cap_drop:" in content
    assert "- ALL" in content
    assert "pids_limit: 256" in content
    assert "mem_limit: 4g" in content
    assert "cpus: 2.0" in content
    assert "/tmp:size=2g,mode=1777" in content


def test_s3_override_installs_optional_dependency_and_selects_backend() -> None:
    content = S3.read_text(encoding="utf-8")
    assert 'INSTALL_S3: "true"' in content
    assert "RESULT_STORAGE_BACKEND: s3" in content


def test_public_preset_binds_only_to_loopback() -> None:
    content = PUBLIC_ENV.read_text(encoding="utf-8")
    assert "API_BIND_ADDRESS=127.0.0.1" in content
    assert "WEB_BIND_ADDRESS=127.0.0.1" in content
    assert "RATE_LIMIT_REQUESTS=10" in content
    assert "RATE_LIMIT_TRUST_PROXY_HEADERS=true" in content


def test_local_preset_remains_local_first() -> None:
    content = LOCAL_ENV.read_text(encoding="utf-8")
    assert "RESULT_STORAGE_BACKEND=filesystem" in content
    assert "RATE_LIMIT_REQUESTS=0" in content


def test_s3_preset_has_no_real_credentials() -> None:
    content = S3_ENV.read_text(encoding="utf-8")
    assert "RESULT_STORAGE_BACKEND=s3" in content
    assert "RESULT_STORAGE_S3_BUCKET=change-me" in content
    assert "AWS_ACCESS_KEY_ID=" in content
    assert "AWS_SECRET_ACCESS_KEY=" in content
    assert "AWS_ACCESS_KEY_ID=AKIA" not in content
