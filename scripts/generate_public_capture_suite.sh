#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
IMAGE=${CAPTURE_TRAIN_IMAGE:-lao-document-ocr-train:latest}
OUTPUT_REL=${CAPTURE_SUITE_OUTPUT:-benchmarks/capture-packs/project-authored-lao-v1}
CORPUS_REL=resources/corpora/project-authored-lao-v1.txt
SUITE_ID=project-authored-lao-v1
COLLECTOR_REL=${CAPTURE_KIT_OUTPUT:-benchmarks/capture-packs/project-authored-lao-v1.collector.zip}

PHETSARATH_VERSION=4.103
PHETSARATH_URL=https://phetsarath.mts.la/downloads/PhetsarathOT-v4.103.zip
PHETSARATH_ZIP_SHA256=a97a317eb1e95c5338a38233176189f63e5b23f29a19630ec317aedee0ce8381
PHETSARATH_REGULAR_SHA256=8dd0fa55de186b051433255d80217007b4026dfc38e2781659424ad7058bbd6e
FONT_CACHE_REL=benchmarks/private/fonts/PhetsarathOT-v4.103
FONT_REL=$FONT_CACHE_REL/PhetsarathOT-Regular.ttf

cd "$ROOT_DIR"

if [ -z "${CAPTURE_FONT_PATH:-}" ]; then
  echo "Preparing Phetsarath OT v$PHETSARATH_VERSION ..."
  python3 - \
    "$FONT_CACHE_REL" \
    "$PHETSARATH_URL" \
    "$PHETSARATH_ZIP_SHA256" \
    "$PHETSARATH_REGULAR_SHA256" <<'PYFONT'
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

destination = Path(sys.argv[1])
url = sys.argv[2]
expected_zip_sha256 = sys.argv[3]
expected_font_sha256 = sys.argv[4]
archive_path = destination.parent / f"{destination.name}.zip"
font_path = destination / "PhetsarathOT-Regular.ttf"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


destination.parent.mkdir(parents=True, exist_ok=True)
if not archive_path.is_file() or sha256(archive_path) != expected_zip_sha256:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response:
        archive_path.write_bytes(response.read())

actual_archive_sha256 = sha256(archive_path)
if actual_archive_sha256 != expected_zip_sha256:
    raise SystemExit(
        "Phetsarath ZIP SHA-256 mismatch: "
        f"{actual_archive_sha256} != {expected_zip_sha256}"
    )

if not font_path.is_file() or sha256(font_path) != expected_font_sha256:
    if destination.exists():
        import shutil

        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)

if not font_path.is_file():
    raise SystemExit(f"Phetsarath font missing after extraction: {font_path}")
actual_font_sha256 = sha256(font_path)
if actual_font_sha256 != expected_font_sha256:
    raise SystemExit(
        "Phetsarath Regular SHA-256 mismatch: "
        f"{actual_font_sha256} != {expected_font_sha256}"
    )

print(f"Phetsarath OT verified: {font_path}")
PYFONT
  FONT_PATH="/workspace/$FONT_REL"
else
  FONT_PATH=$CAPTURE_FONT_PATH
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Building trainer image $IMAGE ..."
  docker build -f Dockerfile.train -t "$IMAGE" .
fi

rm -rf "$OUTPUT_REL"

echo "Generating public capture suite from $CORPUS_REL ..."
docker run --rm --entrypoint python \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  -e PYTHONPATH=/workspace/src \
  "$IMAGE" \
  -m lao_document_ocr.cli generate-capture-suite \
  --corpus "/workspace/$CORPUS_REL" \
  --output "/workspace/$OUTPUT_REL" \
  --font "$FONT_PATH" \
  --suite-id "$SUITE_ID" \
  --text-license Apache-2.0 \
  --text-provenance \
    'Project-authored Lao capture corpus v1; resources/corpora/project-authored-lao-v1.meta.json' \
  --dpi 150 \
  --lines-per-page 8 \
  --max-pages-per-template 10 \
  --require-complete-font

echo "Verifying combined PDF and worksheet ..."
docker run --rm --entrypoint python \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  "$IMAGE" \
  -c "import csv,json,pymupdf; from pathlib import Path; root=Path('/workspace/$OUTPUT_REL'); suite=json.loads((root/'capture-suite.json').read_text(encoding='utf-8')); assert suite['font']=='PhetsarathOT-Regular.ttf', suite['font']; rows=list(csv.DictReader((root/suite['worksheet']).open(encoding='utf-8',newline=''))); pdf=pymupdf.open(root/suite['combined_pdf']); pages=pdf.page_count; pdf.close(); expected=sum(int(pack['page_count']) for pack in suite['packs']); assert pages==expected==len(rows)==60, (pages, expected, len(rows)); print(f'Capture suite verified: {pages} pages / {len(rows)} worksheet rows / font={suite[\"font\"]}'); print(root/suite['combined_pdf']); print(root/suite['worksheet'])"

REVISION=$(git rev-parse HEAD)
echo "Building collector-safe capture kit ..."
docker run --rm --entrypoint python \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  -e PYTHONPATH=/workspace/src \
  "$IMAGE" \
  -m lao_document_ocr.cli build-capture-kit \
  --suite-manifest "/workspace/$OUTPUT_REL/capture-suite.json" \
  --output "/workspace/$COLLECTOR_REL" \
  --revision "$REVISION"

echo "Verifying collector kit ..."
python3 - "$COLLECTOR_REL" <<'PYVERIFY'
import hashlib
import json
import sys
import zipfile
from pathlib import Path

path = Path(sys.argv[1])
with zipfile.ZipFile(path) as archive:
    names = set(archive.namelist())
    expected = {
        "CAPTURE-INSTRUCTIONS.md",
        "SHA256SUMS",
        "capture-kit.json",
        "capture-worksheet.csv",
        "project-authored-lao-v1.pdf",
    }
    assert names == expected, sorted(names)
    worksheet = archive.read("capture-worksheet.csv").decode("utf-8")
    assert "ground_truth" not in worksheet
    assert "digital_page" not in worksheet
    assert "pack_manifest" not in worksheet
    manifest = json.loads(archive.read("capture-kit.json"))
    assert manifest["page_count"] == 60
    assert manifest["font"] == "PhetsarathOT-Regular.ttf"
    assert manifest["excludes_ground_truth"] is True
    assert manifest["excludes_digital_page_images"] is True
    assert manifest["excludes_internal_suite_paths"] is True
    checksum_lines = archive.read("SHA256SUMS").decode("utf-8").splitlines()
    for line in checksum_lines:
        digest, name = line.split("  ", 1)
        assert digest == hashlib.sha256(archive.read(name)).hexdigest()

digest = hashlib.sha256(path.read_bytes()).hexdigest()
print(f"Collector kit verified: {path}")
print(f"SHA-256: {digest}")
PYVERIFY
