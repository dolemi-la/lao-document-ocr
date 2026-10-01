"""Bounded read-only multipart listing for an explicitly selected S3 namespace.

Listing is not evidence of abandonment or permission to abort an upload. No
storage adapter, job manager, object/part reader, or cleanup action is used here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from urllib.parse import unquote_to_bytes, urlsplit

SCHEMA = "s3-multipart-inventory/v1"
MAX_KEY_BYTES = 1024
MAX_UPLOAD_ID_BYTES = 2048
PAGE_SIZE = 1000
_BUCKET = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]\Z")
_BAD_ESCAPE = re.compile(r"%(?![0-9a-fA-F]{2})")


class InventoryError(RuntimeError):
    """Fixed codes only, without provider bodies, identities, paths, or argv."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class InventoryLimits:
    max_uploads: int = 1000
    max_pages: int = 10
    max_items: int = 200
    max_seconds: float = 10.0

    def validate(self) -> None:
        for value, low, high in (
            (self.max_uploads, 1, 100000),
            (self.max_pages, 1, 100),
            (self.max_items, 0, 2000),
        ):
            if type(value) is not int or not low <= value <= high:
                raise InventoryError("invalid_limits")
        if type(self.max_seconds) not in (int, float) or not 0 < self.max_seconds <= 60:
            raise InventoryError("invalid_limits")


def _text(value: object, max_bytes: int, *, controls: bool = False) -> str:
    if not isinstance(value, str) or not 0 < len(value) <= max_bytes:
        raise InventoryError("invalid_response")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError:
        raise InventoryError("invalid_response") from None
    if len(encoded) > max_bytes or (
        not controls and any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise InventoryError("invalid_response")
    return value


def _decoded_key(value: object) -> str:
    # EncodingType=url is explicit in every request. Decode exactly once: '+',
    # literal '%2F', Unicode, and control bytes must never change identities twice.
    if (
        not isinstance(value, str)
        or not 0 < len(value) <= 3 * MAX_KEY_BYTES
        or not value.isascii()
        or _BAD_ESCAPE.search(value)
    ):
        raise InventoryError("invalid_response")
    try:
        decoded = unquote_to_bytes(value).decode("utf-8")
    except (UnicodeError, ValueError):
        raise InventoryError("invalid_response") from None
    return _text(decoded, MAX_KEY_BYTES, controls=True)


def _validate_target(bucket: str, prefix: str, expected_owner: str | None) -> None:
    # Deliberately support named general-purpose buckets, not ARNs/access points
    # or directory buckets (which use a different continuation protocol).
    if (
        not isinstance(bucket, str)
        or not _BUCKET.fullmatch(bucket)
        or any(value in bucket for value in ("..", ".-", "-."))
        or bucket.endswith(("--x-s3", "-s3alias", "--ol-s3", ".mrap", "--table-s3"))
        or re.fullmatch(r"[0-9]+(?:\.[0-9]+){3}", bucket)
    ):
        raise InventoryError("invalid_target")
    try:
        _text(prefix, MAX_KEY_BYTES)
    except InventoryError:
        raise InventoryError("invalid_target") from None
    if (
        not prefix.endswith("/")
        or "\\" in prefix
        or any(part in {"", ".", ".."} for part in prefix[:-1].split("/"))
    ):
        raise InventoryError("invalid_target")
    if expected_owner is not None and (
        not isinstance(expected_owner, str) or not re.fullmatch(r"[0-9]{12}", expected_owner)
    ):
        raise InventoryError("invalid_target")


def _arguments(bucket, prefix, expected_owner, limits, include_keys) -> InventoryLimits:
    limits = InventoryLimits() if limits is None else limits
    if not isinstance(limits, InventoryLimits):
        raise InventoryError("invalid_limits")
    limits.validate()
    if type(include_keys) is not bool:
        raise InventoryError("invalid_arguments")
    _validate_target(bucket, prefix, expected_owner)
    return limits


def _timestamp(value: object) -> str:
    try:
        if not isinstance(value, datetime) or value.utcoffset() is None:
            raise InventoryError("invalid_response")
        return value.astimezone(UTC).isoformat()
    except (ValueError, OverflowError):
        raise InventoryError("invalid_response") from None


def _ref(*values: str) -> str:
    payload = json.dumps(values, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _page(response, *, bucket, prefix, maximum, seen, markers, include_keys, check_time):
    # Validate the entire bounded page before exposing any names or accepting
    # counts. A foreign/malformed entry must not become a partial trusted page.
    if not isinstance(response, dict) or (
        response.get("Bucket") != bucket
        or response.get("EncodingType") != "url"
        or type(response.get("IsTruncated")) is not bool
        or response.get("Delimiter", "") != ""
        or response.get("CommonPrefixes", []) != []
    ):
        raise InventoryError("invalid_response")
    if _decoded_key(response.get("Prefix")) != prefix:
        raise InventoryError("invalid_response")
    metadata = response.get("ResponseMetadata", {})
    if not isinstance(metadata, dict) or metadata.get("HTTPStatusCode", 200) != 200:
        raise InventoryError("invalid_response")
    uploads = response.get("Uploads", [])
    if not isinstance(uploads, list) or len(uploads) > maximum:
        raise InventoryError("invalid_response")
    continuation = None
    if response["IsTruncated"]:
        key = _decoded_key(response.get("NextKeyMarker"))
        upload_id = _text(response.get("NextUploadIdMarker"), MAX_UPLOAD_ID_BYTES)
        if not key.startswith(prefix):
            raise InventoryError("invalid_response")
        continuation = (key, upload_id)
        if continuation in markers:
            raise InventoryError("pagination_cycle")
    items, page_seen = [], set()
    for upload in uploads:
        check_time()
        if not isinstance(upload, dict):
            raise InventoryError("invalid_response")
        key = _decoded_key(upload.get("Key"))
        if not key.startswith(prefix):
            raise InventoryError("invalid_response")
        upload_id = _text(upload.get("UploadId"), MAX_UPLOAD_ID_BYTES)
        initiated_at = _timestamp(upload.get("Initiated"))
        reference = _ref(key, upload_id)
        if reference in seen or reference in page_seen:
            raise InventoryError("duplicate_upload")
        page_seen.add(reference)
        item = {
            "key_ref": _ref(key),
            "upload_ref": reference,
            "initiated_at": initiated_at,
        }
        if include_keys:
            item["key"] = key
        items.append(item)
    check_time()
    return items, page_seen, continuation


def inventory(
    client,
    *,
    bucket: str,
    prefix: str,
    expected_bucket_owner: str | None = None,
    limits: InventoryLimits | None = None,
    include_keys: bool = False,
) -> dict:
    """List metadata only; caller owns its client, credential and timeout policy.

    The returned counts cover accepted pages, not a snapshot, stored bytes, active
    transfer state or abandoned uploads. A malformed page is not partly accepted.
    """
    limits = _arguments(bucket, prefix, expected_bucket_owner, limits, include_keys)
    started = time.monotonic()
    requested = received = accepted = observed = 0
    items: list[dict] = []
    issues: dict[str, int] = {}
    seen: set[str] = set()
    markers: set[tuple[str, str]] = set()
    continuation = None
    finished = False

    def check_time() -> None:
        if time.monotonic() - started >= limits.max_seconds:
            raise InventoryError("time_limit")

    while not finished:
        try:
            check_time()
            if observed >= limits.max_uploads:
                raise InventoryError("upload_limit")
            if requested >= limits.max_pages:
                raise InventoryError("page_limit")
        except InventoryError as exc:
            issues[exc.code] = 1
            break
        maximum = min(PAGE_SIZE, limits.max_uploads - observed)
        arguments = {
            "Bucket": bucket,
            "Prefix": prefix,
            "MaxUploads": maximum,
            "EncodingType": "url",
        }
        if expected_bucket_owner is not None:
            arguments["ExpectedBucketOwner"] = expected_bucket_owner
        if continuation is not None:
            arguments.update(KeyMarker=continuation[0], UploadIdMarker=continuation[1])
        requested += 1
        try:
            response = client.list_multipart_uploads(**arguments)
        except Exception:
            # Do not echo provider codes/messages, account IDs, URLs, or tracebacks.
            issues["listing_unavailable"] = 1
            break
        received += 1
        try:
            check_time()
            page_items, page_seen, next_marker = _page(
                response,
                bucket=bucket,
                prefix=prefix,
                maximum=maximum,
                seen=seen,
                markers=markers,
                include_keys=include_keys,
                check_time=check_time,
            )
            # General-purpose bucket key markers cannot move backwards. Upload
            # IDs remain opaque: distinct IDs for one key may have any ordering.
            if (
                continuation is not None
                and next_marker is not None
                and next_marker[0].encode("utf-8") < continuation[0].encode("utf-8")
            ):
                raise InventoryError("pagination_regression")
        except InventoryError as exc:
            issues[exc.code] = 1
            break
        seen.update(page_seen)
        observed += len(page_items)
        accepted += 1
        items.extend(page_items[: max(0, limits.max_items - len(items))])
        continuation = next_marker
        finished = continuation is None
        if continuation is not None:
            markers.add(continuation)

    omitted = observed - len(items)
    return {
        "schema": SCHEMA,
        "mode": "read-only-multipart-metadata",
        "status": "complete" if finished else "partial",
        "listing_complete": finished,
        "snapshot": False,
        "ownership_verified": False,
        "abandonment_verified": False,
        "deletion_authorized": False,
        "parts_inspected": False,
        "size_bytes": None,
        "limits": asdict(limits),
        "elapsed_seconds": round(max(0, time.monotonic() - started), 6),
        "pages_requested": requested,
        "pages_received": received,
        "pages_accepted": accepted,
        "uploads_observed": observed,
        "issues": issues,
        "details_complete": omitted == 0,
        "items_omitted": omitted,
        "keys_included": include_keys,
        "expected_owner_requested": expected_bucket_owner is not None,
        "selection_order": "provider-listing",
        "items": sorted(items, key=lambda item: item["upload_ref"]),
    }


def _connection_arguments(endpoint_url, region, force_path_style) -> None:
    if type(force_path_style) is not bool:
        raise InventoryError("invalid_arguments")
    if region is not None and (
        not isinstance(region, str) or not re.fullmatch(r"[a-z0-9-]{1,64}", region)
    ):
        raise InventoryError("invalid_arguments")
    if endpoint_url is not None:
        try:
            if not isinstance(endpoint_url, str) or len(endpoint_url) > 2048:
                raise ValueError()
            if any(ord(char) <= 32 or ord(char) == 127 for char in endpoint_url):
                raise ValueError()
            parsed = urlsplit(endpoint_url)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
                or parsed.path not in {"", "/"}
                or parsed.port == 0
            ):
                raise ValueError()
        except (ValueError, TypeError):
            raise InventoryError("invalid_arguments") from None


def build_client(*, endpoint_url=None, region=None, force_path_style=False):
    """Lazy optional SDK import. This is separate from production S3 configuration."""
    _connection_arguments(endpoint_url, region, force_path_style)
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise InventoryError("s3_dependencies_unavailable") from None
    try:
        config = Config(
            connect_timeout=3,
            read_timeout=5,
            max_pool_connections=1,
            retries={"mode": "standard", "total_max_attempts": 1},
            ignore_configured_endpoint_urls=True,
            s3={"addressing_style": "path" if force_path_style else "auto"},
        )
        return boto3.client("s3", endpoint_url=endpoint_url, region_name=region, config=config)
    except Exception:
        raise InventoryError("client_unavailable") from None


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise InventoryError("invalid_arguments")


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--bucket", required=True, help="Explicit general-purpose bucket name.")
    parser.add_argument("--prefix", required=True, help="Nonempty literal prefix ending in '/'.")
    parser.add_argument("--expected-bucket-owner")
    parser.add_argument(
        "--endpoint-url", help="Explicit HTTPS S3-compatible endpoint, without credentials."
    )
    parser.add_argument("--region")
    parser.add_argument("--force-path-style", action="store_true")
    parser.add_argument("--max-uploads", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=10)
    parser.add_argument("--max-items", type=int, default=200)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument(
        "--include-keys", action="store_true", help="Include private object keys, never upload IDs."
    )
    client = None
    try:
        args = parser.parse_args(argv)
        limits = _arguments(
            args.bucket,
            args.prefix,
            args.expected_bucket_owner,
            InventoryLimits(args.max_uploads, args.max_pages, args.max_items, args.max_seconds),
            args.include_keys,
        )
        client = build_client(
            endpoint_url=args.endpoint_url,
            region=args.region,
            force_path_style=args.force_path_style,
        )
        report = inventory(
            client,
            bucket=args.bucket,
            prefix=args.prefix,
            limits=limits,
            include_keys=args.include_keys,
            expected_bucket_owner=args.expected_bucket_owner,
        )
    except InventoryError as exc:
        print(json.dumps({"schema": SCHEMA, "status": "error", "error": exc.code}, sort_keys=True))
        return 2
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                # Never disclose an SDK teardown body or replace useful evidence.
                pass
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2))
    return 0 if report["listing_complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
