from pathlib import Path

CONFIG = Path(__file__).parents[1] / "apps" / "web" / "nginx.conf"


def test_nginx_security_headers_are_configured() -> None:
    content = CONFIG.read_text(encoding="utf-8")
    assert "server_tokens off;" in content
    assert 'X-Content-Type-Options "nosniff"' in content
    assert 'Referrer-Policy "no-referrer"' in content
    assert 'X-Frame-Options "DENY"' in content
    assert "Permissions-Policy" in content
    assert "Content-Security-Policy" in content
    assert "frame-ancestors 'none'" in content
    assert "object-src 'none'" in content
