from __future__ import annotations

import argparse
from pathlib import Path

from lao_document_ocr.corpus import (
    CorpusFilter,
    iter_jsonl,
    iter_plain_text,
    prepare_corpus,
    write_corpus,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a normalized, filtered Lao text corpus for synthetic OCR generation."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--format", choices=["text", "jsonl"], default="text")
    parser.add_argument("--field", default="text", help="JSONL text field")
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=180)
    parser.add_argument("--min-lao-ratio", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--keep-duplicates", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.format == "jsonl":
        source = iter_jsonl(args.input, field=args.field)
    else:
        source = iter_plain_text(args.input)

    config = CorpusFilter(
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_lao_ratio=args.min_lao_ratio,
        deduplicate=not args.keep_duplicates,
    )
    lines = prepare_corpus(source, config, limit=args.limit)
    write_corpus(lines, args.output)

    print(f"Wrote {len(lines)} lines to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
