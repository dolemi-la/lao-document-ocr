#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
IMAGE=${CAPTURE_TRAIN_IMAGE:-lao-document-ocr-train:latest}
OUTPUT_REL=${CAPTURE_SUITE_OUTPUT:-benchmarks/capture-packs/project-authored-lao-v1}
CORPUS_REL=resources/corpora/project-authored-lao-v1.txt
FONT_PATH=${CAPTURE_FONT_PATH:-/usr/share/fonts/truetype/noto/NotoSansLao-Regular.ttf}
SUITE_ID=project-authored-lao-v1
COLLECTOR_REL=${CAPTURE_KIT_OUTPUT:-benchmarks/capture-packs/project-authored-lao-v1.collector.zip}

cd "$ROOT_DIR"

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
  --max-pages-per-template 10

echo "Verifying combined PDF and worksheet ..."
docker run --rm --entrypoint python \
  -v "$ROOT_DIR:/workspace" \
  -w /workspace \
  "$IMAGE" \
  -c "import csv,json,pymupdf; from pathlib import Path; root=Path('/workspace/$OUTPUT_REL'); suite=json.loads((root/'capture-suite.json').read_text(encoding='utf-8')); rows=list(csv.DictReader((root/suite['worksheet']).open(encoding='utf-8',newline=''))); pdf=pymupdf.open(root/suite['combined_pdf']); pages=pdf.page_count; pdf.close(); expected=sum(int(pack['page_count']) for pack in suite['packs']); assert pages==expected==len(rows)==60, (pages, expected, len(rows)); print(f'Capture suite verified: {pages} pages / {len(rows)} worksheet rows'); print(root/suite['combined_pdf']); print(root/suite['worksheet'])"


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
