#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
IMAGE=${CAPTURE_TRAIN_IMAGE:-lao-document-ocr-train:latest}
OUTPUT_REL=${CAPTURE_SUITE_OUTPUT:-benchmarks/capture-packs/project-authored-lao-v1}
CORPUS_REL=resources/corpora/project-authored-lao-v1.txt
FONT_PATH=${CAPTURE_FONT_PATH:-/usr/share/fonts/truetype/noto/NotoSansLao-Regular.ttf}
SUITE_ID=project-authored-lao-v1

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
