"""Actual botocore request/XML path with fake credentials and an in-memory transport.

Install .[s3] to run these locally; CI's Python job includes that optional extra.
No S3 request is sent, including failure/retry tests.
"""

from __future__ import annotations

import socket
from urllib.parse import parse_qs, quote, urlsplit
from xml.sax.saxutils import escape

import pytest
from test_s3_multipart_inventory import BUCKET, PREFIX

from services.api.app.s3_multipart_inventory import build_client, inventory


@pytest.fixture
def sdk_client(monkeypatch, tmp_path):
    boto3 = pytest.importorskip("boto3", reason="Install the optional .[s3] extra")
    monkeypatch.setattr(boto3, "DEFAULT_SESSION", None)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "authored-fixture-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "authored-fixture-secret")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "absent-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "absent-credentials"))
    for name in (
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_SESSION_TOKEN",
        "AWS_ROLE_ARN",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    ):
        monkeypatch.delenv(name, raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("The real SDK test must never connect to a network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    client = build_client(endpoint_url="https://storage.invalid", region="us-east-1")
    try:
        yield client
    finally:
        client.close()


class RawResponse:
    def __init__(self, body):
        self.body = body

    def stream(self, amt=None, decode_content=False):
        yield self.body


def test_real_sdk_encoded_xml_and_request_markers_preserve_key_identity(sdk_client, monkeypatch):
    from botocore.awsrequest import AWSResponse

    key = PREFIX + "ລາວ + %2F\n.zip"
    calls = []

    def send(request):
        calls.append(parse_qs(urlsplit(request.url).query))
        first = len(calls) == 1
        upload_id = "opaque-Z+/=" if first else "opaque-A+/="
        markers = (
            (
                f"<NextKeyMarker>{escape(quote(key, safe='/'))}</NextKeyMarker>"
                f"<NextUploadIdMarker>{escape(upload_id)}</NextUploadIdMarker>"
            )
            if first
            else ""
        )
        body = (
            '<ListMultipartUploadsResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            f"<Bucket>{BUCKET}</Bucket><Prefix>{escape(quote(PREFIX, safe='/'))}</Prefix>"
            "<EncodingType>url</EncodingType>"
            f"<IsTruncated>{str(first).lower()}</IsTruncated>{markers}"
            f"<Upload><Key>{escape(quote(key, safe='/'))}</Key>"
            f"<UploadId>{escape(upload_id)}</UploadId>"
            "<Initiated>2026-01-01T00:00:00.000Z</Initiated></Upload>"
            "</ListMultipartUploadsResult>"
        ).encode()
        return AWSResponse(request.url, 200, {"content-type": "application/xml"}, RawResponse(body))

    monkeypatch.setattr(sdk_client._endpoint.http_session, "send", send)
    report = inventory(sdk_client, bucket=BUCKET, prefix=PREFIX, include_keys=True)
    assert report["listing_complete"], report["issues"]
    assert report["uploads_observed"] == 2
    assert {item["key"] for item in report["items"]} == {key}
    assert calls[0]["encoding-type"] == ["url"]
    assert calls[1]["key-marker"] == [key]
    assert calls[1]["upload-id-marker"] == ["opaque-Z+/="]
    assert all(call["prefix"] == [PREFIX] for call in calls)
    assert len(calls) == 2


def test_real_sdk_provider_failure_has_no_automatic_retry_or_body_leak(sdk_client, monkeypatch):
    from botocore.awsrequest import AWSResponse

    calls = []

    def send(request):
        calls.append(request.method)
        body = b"<Error><Code>SlowDown</Code><Message>PRIVATE provider body</Message></Error>"
        return AWSResponse(request.url, 503, {"content-type": "application/xml"}, RawResponse(body))

    monkeypatch.setattr(sdk_client._endpoint.http_session, "send", send)
    report = inventory(sdk_client, bucket=BUCKET, prefix=PREFIX)
    assert report["issues"] == {"listing_unavailable": 1}
    assert report["pages_requested"] == 1
    assert calls == ["GET"]
    assert "PRIVATE" not in str(report)
    assert sdk_client.meta.config.retries["total_max_attempts"] == 1
    assert sdk_client.meta.config.connect_timeout == 3
    assert sdk_client.meta.config.read_timeout == 5
