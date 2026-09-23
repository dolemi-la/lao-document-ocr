#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PROJECT_NAME=${SECURITY_SMOKE_PROJECT:-lao-ocr-security-smoke}
API_PORT=${SECURITY_SMOKE_API_PORT:-18000}
WEB_PORT=${SECURITY_SMOKE_WEB_PORT:-15173}

cd "$ROOT_DIR"

compose() {
  API_PORT="$API_PORT" \
  WEB_PORT="$WEB_PORT" \
  RATE_LIMIT_REQUESTS=2 \
  RATE_LIMIT_WINDOW_SECONDS=60 \
  docker compose \
    -p "$PROJECT_NAME" \
    -f docker-compose.yml \
    -f deploy/compose.public.yml \
    "$@"
}

cleanup() {
  compose down --volumes --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

fail() {
  echo "SECURITY SMOKE FAILED: $*" >&2
  exit 1
}

wait_for_url() {
  url=$1
  attempts=${2:-60}
  count=0
  while [ "$count" -lt "$attempts" ]; do
    if curl --fail --silent --show-error "$url" >/dev/null 2>&1; then
      return 0
    fi
    count=$((count + 1))
    sleep 1
  done
  return 1
}

header_value() {
  url=$1
  name=$2
  shift 2
  curl --silent --show-error --head "$@" "$url" \
    | tr -d '\r' \
    | awk -F': ' -v wanted="$(printf '%s' "$name" | tr '[:upper:]' '[:lower:]')" '
        tolower($1) == wanted {
          $1="";
          sub(/^:?[[:space:]]*/, "");
          print;
          exit
        }
      '
}

echo "Building hardened API/web images..."
compose build api web

echo "Starting hardened stack..."
compose up -d api web

wait_for_url "http://127.0.0.1:${API_PORT}/health" \
  || fail "API did not become healthy"
wait_for_url "http://127.0.0.1:${WEB_PORT}/" \
  || fail "web UI did not become healthy"

echo "Checking non-root runtime and private writable paths..."
api_uid=$(compose exec -T api id -u | tr -d '\r\n')
[ "$api_uid" = "10001" ] || fail "API uid is $api_uid, expected 10001"

modes=$(compose exec -T api sh -c \
  "stat -c '%a' /data/results /tmp/lao-document-ocr-jobs")
printf '%s\n' "$modes" | awk '
  $1 != "700" { exit 1 }
  END { if (NR != 2) exit 1 }
' || fail "one or more writable API roots are not mode 0700"

echo "Checking API security headers..."
api_url="http://127.0.0.1:${API_PORT}/health"
[ "$(header_value "$api_url" "X-Content-Type-Options")" = "nosniff" ] \
  || fail "API nosniff header missing"
[ "$(header_value "$api_url" "X-Frame-Options")" = "DENY" ] \
  || fail "API frame protection missing"
[ "$(header_value "$api_url" "Referrer-Policy")" = "no-referrer" ] \
  || fail "API referrer policy missing"
header_value "$api_url" "Content-Security-Policy" | grep -q "default-src 'none'" \
  || fail "API CSP missing strict default-src"

request_id=$(header_value "$api_url" "X-Request-ID" -H "X-Request-ID: security-smoke-123")
[ "$request_id" = "security-smoke-123" ] \
  || fail "API request-id propagation failed"

echo "Checking Swagger docs-specific CSP..."
docs_url="http://127.0.0.1:${API_PORT}/docs"
curl --fail --silent --show-error "$docs_url" | grep -q "Swagger UI" \
  || fail "Swagger UI is not usable"
header_value "$docs_url" "Content-Security-Policy" \
  | grep -q "https://cdn.jsdelivr.net" \
  || fail "Swagger CSP does not allow its required CDN"

echo "Checking web security headers..."
web_url="http://127.0.0.1:${WEB_PORT}/"
[ "$(header_value "$web_url" "X-Content-Type-Options")" = "nosniff" ] \
  || fail "web nosniff header missing"
[ "$(header_value "$web_url" "X-Frame-Options")" = "DENY" ] \
  || fail "web frame protection missing"
header_value "$web_url" "Content-Security-Policy" | grep -q "object-src 'none'" \
  || fail "web CSP missing object-src restriction"

tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"; cleanup' EXIT INT TERM

printf 'not a pdf' >"$tmp_dir/fake.pdf"

echo "Checking content/extension validation..."
fake_code=$(curl --silent --show-error \
  --output "$tmp_dir/fake-response.json" \
  --write-out '%{http_code}' \
  --request POST \
  --form "file=@$tmp_dir/fake.pdf;type=application/pdf" \
  "http://127.0.0.1:${API_PORT}/v1/jobs")
[ "$fake_code" = "422" ] \
  || fail "fake PDF returned HTTP $fake_code instead of 422"
grep -q "does not match the .pdf extension" "$tmp_dir/fake-response.json" \
  || fail "fake PDF rejection message missing"

printf '%s' \
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=' \
  | base64 --decode >"$tmp_dir/page.png"

echo "Checking rate limit and polling exemption..."
first_code=$(curl --silent --show-error \
  --output "$tmp_dir/first-job.json" \
  --write-out '%{http_code}' \
  --request POST \
  --form "file=@$tmp_dir/page.png;type=image/png" \
  "http://127.0.0.1:${API_PORT}/v1/jobs")
[ "$first_code" = "202" ] \
  || fail "first valid job returned HTTP $first_code instead of 202"

job_id=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["id"])' \
  "$tmp_dir/first-job.json")

third_code=$(curl --silent --show-error \
  --dump-header "$tmp_dir/limited.headers" \
  --output "$tmp_dir/limited.json" \
  --write-out '%{http_code}' \
  --request POST \
  --form "file=@$tmp_dir/page.png;type=image/png" \
  "http://127.0.0.1:${API_PORT}/v1/jobs")
[ "$third_code" = "429" ] \
  || fail "rate-limited job returned HTTP $third_code instead of 429"
grep -qi '^Retry-After:' "$tmp_dir/limited.headers" \
  || fail "rate-limited response is missing Retry-After"

poll_code=$(curl --silent --show-error \
  --output "$tmp_dir/poll.json" \
  --write-out '%{http_code}' \
  "http://127.0.0.1:${API_PORT}/v1/jobs/${job_id}")
[ "$poll_code" = "200" ] \
  || fail "job polling was incorrectly rate limited (HTTP $poll_code)"

echo "Checking Prometheus metrics and route cardinality..."
metrics=$(curl --fail --silent --show-error \
  "http://127.0.0.1:${API_PORT}/metrics")
printf '%s\n' "$metrics" | grep -q \
  'route="/v1/jobs",status="429"' \
  || fail "429 submission metric missing"
printf '%s\n' "$metrics" | grep -q \
  'route="/v1/jobs/{job_id}"' \
  || fail "normalized job route metric missing"
if printf '%s\n' "$metrics" | grep -q "$job_id"; then
  fail "raw job id leaked into metric labels"
fi

echo "Checking hardened Compose settings..."
compose config >"$tmp_dir/compose.yml"
grep -q 'read_only: true' "$tmp_dir/compose.yml" \
  || fail "public API read_only setting missing"
grep -q 'no-new-privileges:true' "$tmp_dir/compose.yml" \
  || fail "no-new-privileges setting missing"
grep -q 'pids_limit: 256' "$tmp_dir/compose.yml" \
  || fail "PID limit missing"

echo "Security smoke passed."
