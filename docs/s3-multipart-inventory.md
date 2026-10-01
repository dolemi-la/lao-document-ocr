# Bounded read-only S3 multipart inventory

Starting checkout: `7c344642f48ed930c81ac0c6608798e218b8a86a`.

## Scope

This standalone operator command lists metadata for incomplete multipart uploads
under an explicitly selected bucket and nonempty prefix. It does not abort an
upload, delete an object, list parts, download contents, inspect the job journal,
construct a production storage adapter, or start the API. It is not scheduled or
invoked by normal request handling. Production settings are unchanged.

The [filesystem inventory](filesystem-remnant-inventory.md) remains independent.
Neither inventory grants ownership or deletion authority. Multipart initiation
timestamps do not establish abandonment, and a listing cannot prove that a
writer has stopped. Actual cleanup and late-write reconciliation remain open.

## Explicit invocation

From the source checkout with the optional S3 dependencies installed:

```bash
.venv/bin/python -m pip install -e '.[s3]'
.venv/bin/python -B -m services.api.app.s3_multipart_inventory \
  --bucket "${RESULT_STORAGE_S3_BUCKET:?Select the intended bucket}" \
  --prefix 'prod/jobs/' \
  --max-uploads 1000 \
  --max-pages 10 \
  --max-items 200 \
  --max-seconds 10
```

Replace `prod/jobs/` with the exact intended namespace; the command does not
infer or append `jobs/` from production configuration. An empty prefix, `/`,
missing trailing slash, or ambiguous path segment is rejected rather than
expanded into a bucket-wide scan. Prefix matching is literal, not a wildcard.
Bucket names, the prefix, and all limits are validated before SDK construction.

This command makes authenticated listing requests when deliberately invoked.
It is read-only, not network-free or necessarily free of provider request costs.
Use an operator identity with listing permission for the intended bucket, without
abort/delete permissions. See the [AWS multipart permissions reference](https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpuoverview.html#mpuAndPermissions).
No provider credentials are embedded in the command, repository, or report.

For an S3-compatible service, supply `--endpoint-url https://storage.example.com`
and the appropriate `--region` and optional `--force-path-style`. Only explicit
HTTPS base endpoints are accepted; embedded credentials, queries, fragments,
non-root paths, and invalid ports are rejected. TLS verification is not disabled.
Configured SDK endpoint overrides are ignored, preventing an environment/config
endpoint from silently replacing the command's selected default or explicit URL.
Ordinary SDK credential and region resolution still apply.

`--expected-bucket-owner` optionally sends a 12-digit expected account ID on every
page request. The report notes only that the guard was requested; it does not
claim independent account or cleanup ownership verification.

The supported protocol is a named general-purpose bucket. Directory buckets,
access-point/Outposts ARNs, and known special bucket aliases are rejected instead
of assuming their pagination/authentication behavior is interchangeable. A
compatible provider must honor the requested prefix and encoded response shape.

## Output and privacy

JSON goes to standard output. There is no report-file, abort, delete, repair, or
confirmation option. Redirect only into an operator-controlled private location.
The schema is `s3-multipart-inventory/v1`.

Default items contain a key digest, a distinct key/upload-ID-pair digest, and the
observed initiation timestamp normalized to UTC. The bucket, prefix, endpoint,
raw keys, upload IDs, owner/initiator metadata, request IDs, and provider error
bodies are not copied into reports. Provider/argument failures use fixed codes.

`--include-keys` explicitly includes private object keys. Upload IDs and
owner/initiator identities remain excluded even then. JSON escapes control
characters and Unicode rather than emitting raw terminal-control names.
Predictable key hashes are not anonymization; timestamps and correlated digests
are sensitive. Keep target/configuration context separately and privately: a
hash does not authenticate a bucket, endpoint, owner, or file.

`uploads_observed` counts distinct accepted key/upload-ID pairs. Several uploads
of the same key remain distinct. No object-size, stored-byte, part-count, cost,
age-threshold, safe-to-abort, or abandonment estimate is inferred. `size_bytes`
is explicitly null and `parts_inspected` is false.

## Pagination and validation

Every listing requests explicit URL encoding and decodes keys exactly once.
Literal plus signs and percent sequences retain their identity. Both continuation
markers are carried to the next request; upload IDs are opaque and not sorted
lexicographically by the inventory. This follows the documented
[ListMultipartUploads response contract](https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListMultipartUploads.html).

An entire bounded page is validated before its counts or keys are accepted.
Wrong buckets/prefixes, grouped results, absent/invalid encoding, malformed
entries or dates, excessive page length, duplicate upload identities, cyclic
markers, and backwards key markers stop traversal with partial evidence. Valid
pages accepted earlier remain available; the invalid page is not partly trusted.
A listing/access/transport failure is never represented as an empty success.

`listing_complete` means the accepted traversal reached an explicit final page
without local validation or budget failures. It is not a point-in-time snapshot:
provider contents may change during pagination, and this command cannot prove a
provider returned every possible upload. It does not join listings to live job
ownership or confirm whether an upload has since completed.

## Limits and exit codes

| Limit | Default | Accepted range |
| --- | --- | --- |
| Uploads observed | 1,000 | 1–100,000 |
| Listing calls | 10 | 1–100 |
| Detail items retained | 200 | 0–2,000 |
| Cooperative elapsed seconds | 10 | Greater than 0, at most 60 |

Each request asks for at most 1,000 uploads or the remaining upload budget,
whichever is smaller. The exact upload cap can still be complete when the
provider explicitly marks that page final. Otherwise the cap is partial.
Detail truncation is separate: `details_complete` and `items_omitted` refer only
to accepted uploads, not unvisited/invalid pages. `pages_requested` includes
failed listing calls; `pages_received` includes invalid responses;
`pages_accepted` counts validated pages. Details preserve a bounded first-page
selection and are sorted by opaque reference for presentation.

Key bytes and upload-ID bytes are separately bounded. The SDK has already
received/parsed a response before page validation; this is not a raw response
body or hostile-endpoint memory quota. A bounded count of seen identity hashes
is retained for duplicate detection. No paginator auto-fetches further pages.

The standalone SDK client configures 3-second connection and 5-second read
timeouts, with one total configured attempt and a single pooled connection.
See the [botocore configuration reference](https://docs.aws.amazon.com/botocore/latest/reference/config.html).
The cooperative timer starts after client construction and cannot interrupt DNS,
credential discovery/refresh, SDK routing behavior, or a stalled in-flight call.
It is not a hard end-to-end wall-clock or exact network-roundtrip guarantee.
Programmatic `inventory(client, ...)` callers retain responsibility for their
client's credentials, retries, transport limits, and closure.

Exit 0 means listing traversal completed. Exit 3 means partial evidence,
including failed or malformed listing responses even on the first page. Exit 2
means invalid arguments/target, missing optional dependencies, or failed client
initialization. The CLI attempts client closure on every exit after construction;
a close failure is not a cleanup receipt and its private exception is suppressed.

## Verification and remaining work

The 100 new focused cases cover pagination, per-key upload identity, encoded
Unicode/control/percent keys, malformed and foreign pages, duplicate/cyclic/
backwards markers, all budgets, error privacy, CLI isolation, client finalization,
and SDK configuration. Two tests use the real botocore request serializer and XML
parser with fake credentials and an in-memory HTTP transport. Network connections
are forbidden in those tests. The Python CI job installs the existing optional
S3 extra so these two checks run rather than skip. No production dependency
requirement or service configuration is changed.

Self-review reproduced and fixed a backwards-key continuation that could
incorrectly reach a complete result. Tests are synthetic protocol/transport
fixtures, not a live AWS/R2/MinIO compatibility result, real bucket inventory,
container startup, or OCR accuracy measurement. No live storage was listed.

Local plan, red tests, SDK version/install log, focused/full gates, self-review,
and protected hashes are retained under the Git-ignored directory
`benchmarks/private/s3-multipart-inventory-7c34464/`. The isolated SDK test install
does not replace the existing development environment's packages.

The next reconciliation work still requires explicit upload ownership and writer
coordination before introducing any abort policy. This report does not authorize
bulk cleanup, provide automatic lifecycle changes, or reconcile late provider
writes. Legacy filesystem deletion policy and framework spool handling remain
separate open tasks.

## Completed local gates

All 100 focused cases and all 1,751 Python tests passed with the isolated S3 SDK
available, including both real-SDK transport checks. Six existing dependency
deprecation warnings remained, with no test failures or skipped SDK checks.
Ruff, web lint, all 145 web tests, the production web build, local documentation
links, and whitespace validation passed. All 19 protected-input hashes matched
before commit, including Phetsarath, recognition/evaluation assets, the existing
job manager, storage implementation, journal, and filesystem inventory.
