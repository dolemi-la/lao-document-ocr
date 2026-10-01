"""Read-only protocol fixtures; no AWS credentials, SDK, or live account needed."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from urllib.parse import quote

import pytest

from services.api.app.s3_multipart_inventory import InventoryError, InventoryLimits, inventory

BUCKET = "fixture-bucket"
PREFIX = "private-project/jobs/"
INITIATED = datetime(2026, 1, 1, tzinfo=UTC)


def upload(key=PREFIX + "fixture.zip", upload_id="PRIVATE-upload-id", **extra):
    return {"Key": quote(key, safe="/"), "UploadId": upload_id, "Initiated": INITIATED, **extra}


def page(uploads=(), *, more=False, next_key=None, next_id=None, **extra):
    result = {
        "Bucket": BUCKET,
        "Prefix": quote(PREFIX, safe="/"),
        "EncodingType": "url",
        "IsTruncated": more,
        "Uploads": list(uploads),
    }
    if next_key is not None:
        result["NextKeyMarker"] = quote(next_key, safe="/")
    if next_id is not None:
        result["NextUploadIdMarker"] = next_id
    return {**result, **extra}


class ListingClient:
    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    def list_multipart_uploads(self, **kwargs):
        self.calls.append(kwargs)
        assert self.pages, "Unexpected extra provider call"
        result = self.pages.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def __getattr__(self, name):
        raise AssertionError(f"Only multipart listing is allowed, not {name}")


def run(client, **kwargs):
    return inventory(client, bucket=BUCKET, prefix=PREFIX, **kwargs)


def test_empty_listing_is_complete_but_not_ownership_evidence():
    client = ListingClient([page()])
    report = run(client)
    assert report["listing_complete"] is True
    assert report["uploads_observed"] == 0
    assert report["status"] == "complete"
    assert report["issues"] == {}
    assert report["snapshot"] is False
    assert report["ownership_verified"] is False
    assert report["abandonment_verified"] is False
    assert report["deletion_authorized"] is False
    assert report["parts_inspected"] is False
    assert report["size_bytes"] is None
    assert client.calls == [
        {
            "Bucket": BUCKET,
            "Prefix": PREFIX,
            "MaxUploads": 1000,
            "EncodingType": "url",
        }
    ]


def test_both_markers_are_preserved_for_multiple_uploads_of_the_same_key():
    key = PREFIX + "ລາວ + %2F.zip"
    responses = [
        page([upload(key, "opaque-Z+/=")], more=True, next_key=key, next_id="opaque-Z+/="),
        page([upload(key, "opaque-A+/=")]),
    ]
    before = copy.deepcopy(responses)
    client = ListingClient(responses)
    report = run(client, include_keys=True, expected_bucket_owner="123456789012")
    assert report["uploads_observed"] == 2
    assert report["listing_complete"]
    assert {item["key"] for item in report["items"]} == {key}
    assert len({item["upload_ref"] for item in report["items"]}) == 2
    assert len({item["key_ref"] for item in report["items"]}) == 1
    assert client.calls[1]["KeyMarker"] == key
    assert client.calls[1]["UploadIdMarker"] == "opaque-Z+/="
    assert all(call["ExpectedBucketOwner"] == "123456789012" for call in client.calls)
    assert all(call["Prefix"] == PREFIX for call in client.calls)
    assert responses == before  # Do not mutate provider data while decoding.


def test_default_report_excludes_keys_ids_owners_and_provider_metadata():
    data = page(
        [
            upload(
                PREFIX + "PRIVATE-ລາວ\n.zip",
                Owner={"ID": "PRIVATE-owner"},
                Initiator={"DisplayName": "PRIVATE-user"},
                StorageClass="PRIVATE-storage",
            )
        ],
        ResponseMetadata={"HTTPStatusCode": 200, "RequestId": "PRIVATE-request"},
    )
    report = run(ListingClient([data]))
    serialized = json.dumps(report)
    assert "PRIVATE" not in serialized
    assert BUCKET not in serialized
    assert PREFIX not in serialized
    assert "ລາວ" not in serialized
    assert report["items"][0]["initiated_at"] == INITIATED.isoformat()
    assert set(report["items"][0]) == {"key_ref", "upload_ref", "initiated_at"}
    assert report["keys_included"] is False


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, 100001])
def test_upload_limit_validated_before_listing(limit):
    client = ListingClient([])
    with pytest.raises(InventoryError, match="invalid_limits"):
        run(client, limits=replace(InventoryLimits(), max_uploads=limit))
    assert not client.calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_pages", 0),
        ("max_pages", True),
        ("max_pages", 101),
        ("max_items", -1),
        ("max_items", True),
        ("max_items", 2001),
        ("max_seconds", 0),
        ("max_seconds", True),
        ("max_seconds", float("nan")),
        ("max_seconds", float("inf")),
        ("max_seconds", 61),
    ],
)
def test_other_limits_are_bounded(field, value):
    with pytest.raises(InventoryError, match="invalid_limits"):
        run(ListingClient([]), limits=replace(InventoryLimits(), **{field: value}))


@pytest.mark.parametrize("prefix", ["", "/", "jobs", "jobs//", "../jobs/", "/jobs/", "jobs\n/"])
def test_explicit_narrow_prefix_is_required(prefix):
    client = ListingClient([])
    with pytest.raises(InventoryError, match="invalid_target"):
        inventory(client, bucket=BUCKET, prefix=prefix)
    assert not client.calls


@pytest.mark.parametrize("bucket", ["", "ab", "arn:aws:s3:PRIVATE", "a/b", "a--zone--x-s3"])
def test_unsupported_bucket_forms_fail_before_listing(bucket):
    client = ListingClient([])
    with pytest.raises(InventoryError, match="invalid_target"):
        inventory(client, bucket=bucket, prefix=PREFIX)
    assert not client.calls


def test_page_cap_stops_even_when_provider_has_more():
    client = ListingClient([page([], more=True, next_key=PREFIX + "a", next_id="a")])
    report = run(client, limits=replace(InventoryLimits(), max_pages=1))
    assert report["status"] == "partial"
    assert report["issues"] == {"page_limit": 1}
    assert report["pages_requested"] == 1
    assert not report["listing_complete"]


def test_upload_cap_limits_request_size_and_stops_before_second_page():
    client = ListingClient([page([upload()], more=True, next_key=PREFIX + "a", next_id="a")])
    report = run(client, limits=replace(InventoryLimits(), max_uploads=1))
    assert report["uploads_observed"] == 1
    assert report["issues"] == {"upload_limit": 1}
    assert client.calls[0]["MaxUploads"] == 1


def test_exact_upload_cap_is_complete_when_provider_explicitly_finishes():
    report = run(
        ListingClient([page([upload()])]), limits=replace(InventoryLimits(), max_uploads=1)
    )
    assert report["listing_complete"]
    assert report["issues"] == {}


def test_detail_cap_is_not_listing_truncation():
    rows = [upload(PREFIX + str(i), str(i)) for i in range(4)]
    report = run(ListingClient([page(rows)]), limits=replace(InventoryLimits(), max_items=1))
    assert report["listing_complete"]
    assert report["uploads_observed"] == 4
    assert report["items_omitted"] == 3
    assert not report["details_complete"]
    assert len(report["items"]) == 1


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        {},
        page(IsTruncated="false"),
        page(EncodingType="invalid"),
        page(Bucket="foreign-bucket"),
        page(Prefix="foreign/"),
        page(Uploads={}),
        page(CommonPrefixes=[{"Prefix": PREFIX}]),
        page(Delimiter="/"),
        page(ResponseMetadata={"HTTPStatusCode": 403}),
    ],
)
def test_malformed_or_foreign_page_is_partial_not_empty_success(bad):
    report = run(ListingClient([bad]), include_keys=True)
    assert not report["listing_complete"]
    assert report["issues"] == {"invalid_response": 1}
    assert report["pages_accepted"] == 0
    assert report["uploads_observed"] == 0
    assert report["items"] == []


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        upload("foreign/private.zip"),
        upload(UploadId=""),
        upload(UploadId="\nPRIVATE"),
        upload(Initiated="PRIVATE-date"),
        upload(Initiated=datetime(2026, 1, 1)),
        upload(Key="broken%XX"),
        upload(Key=PREFIX + "%FF"),
        upload(Key=PREFIX + "x" * 1024),
    ],
)
def test_invalid_upload_rejects_whole_page_without_exposing_partial_names(bad):
    report = run(ListingClient([page([upload(), bad])]), include_keys=True)
    assert report["issues"] == {"invalid_response": 1}
    assert report["uploads_observed"] == 0
    assert not report["items"]


def test_oversized_page_is_rejected_without_ignoring_extra_entries():
    client = ListingClient([page([upload(), upload(PREFIX + "other", "other")])])
    report = run(client, limits=replace(InventoryLimits(), max_uploads=1))
    assert report["issues"] == {"invalid_response": 1}
    assert report["uploads_observed"] == 0


@pytest.mark.parametrize("same_page", [True, False])
def test_duplicate_key_and_upload_id_does_not_inflate_observations(same_page):
    responses = (
        [page([upload(), upload()])]
        if same_page
        else [
            page([upload()], more=True, next_key=PREFIX + "a", next_id="a"),
            page([upload()]),
        ]
    )
    report = run(ListingClient(responses))
    assert not report["listing_complete"]
    assert report["issues"] == {"duplicate_upload": 1}
    assert report["uploads_observed"] == (0 if same_page else 1)


@pytest.mark.parametrize(
    "next_key,next_id", [(None, None), (PREFIX + "a", None), ("foreign/a", "a")]
)
def test_bad_continuation_is_not_followed(next_key, next_id):
    client = ListingClient([page([], more=True, next_key=next_key, next_id=next_id)])
    report = run(client)
    assert report["issues"] == {"invalid_response": 1}
    assert len(client.calls) == 1


def test_cyclic_continuation_stops_without_another_request():
    client = ListingClient(
        [
            page([], more=True, next_key=PREFIX + "a", next_id="a"),
            page([], more=True, next_key=PREFIX + "b", next_id="b"),
            page([], more=True, next_key=PREFIX + "a", next_id="a"),
        ]
    )
    report = run(client)
    assert report["issues"] == {"pagination_cycle": 1}
    assert len(client.calls) == 3


@pytest.mark.parametrize("after_success", [False, True])
def test_provider_errors_are_private_and_preserve_only_prior_accepted_pages(after_success, caplog):
    responses = [OSError("PRIVATE credentials, request body, endpoint")]
    if after_success:
        responses.insert(0, page([upload()], more=True, next_key=PREFIX + "a", next_id="a"))
    report = run(ListingClient(responses))
    assert report["issues"] == {"listing_unavailable": 1}
    assert report["uploads_observed"] == int(after_success)
    assert not report["listing_complete"]
    assert "PRIVATE" not in json.dumps(report) + caplog.text


def test_cooperative_deadline_stops_after_blocked_request_returns(monkeypatch):
    import services.api.app.s3_multipart_inventory as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    client = ListingClient([])

    def late(**kwargs):
        client.calls.append(kwargs)
        clock[0] = 20.0
        return page([upload()])

    monkeypatch.setattr(client, "list_multipart_uploads", late)
    report = run(client, limits=replace(InventoryLimits(), max_seconds=1))
    assert report["issues"] == {"time_limit": 1}
    assert report["uploads_observed"] == 0
    assert len(client.calls) == 1


def test_initiation_timestamp_is_normalized_without_age_judgment():
    zone = datetime(2026, 1, 1, 7, tzinfo=timezone(timedelta(hours=7)))
    report = run(ListingClient([page([upload(Initiated=zone)])]))
    assert report["items"][0]["initiated_at"] == INITIATED.isoformat()
    assert not {"age_seconds", "stale", "abandoned"} & report.keys()
    assert not {"age_seconds", "stale", "abandoned"} & report["items"][0].keys()


def test_backward_key_marker_cannot_be_reported_as_a_complete_listing():
    client = ListingClient(
        [
            page([], more=True, next_key=PREFIX + "z", next_id="first"),
            page([], more=True, next_key=PREFIX + "a", next_id="different"),
            page(),
        ]
    )
    report = run(client)
    assert report["issues"] == {"pagination_regression": 1}
    assert not report["listing_complete"]
    assert len(client.calls) == 2
