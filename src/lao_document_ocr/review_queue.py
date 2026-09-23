# ruff: noqa: E501
from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from lao_document_ocr.dataset import (
    DatasetReviewStatus,
    DatasetSample,
    DatasetSplit,
)


@dataclass(frozen=True)
class ReviewQueueEntry:
    id: str
    document_id: str
    split: str
    subset: str
    tags: tuple[str, ...]
    source: str
    ground_truth: str
    review_status: str
    reviewer: str | None
    review_notes: str | None
    thumbnail: str | None
    ground_truth_text: str | None
    problems: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "split": self.split,
            "subset": self.subset,
            "tags": list(self.tags),
            "source": self.source,
            "ground_truth": self.ground_truth,
            "review_status": self.review_status,
            "reviewer": self.reviewer,
            "review_notes": self.review_notes,
            "thumbnail": self.thumbnail,
            "ground_truth_text": self.ground_truth_text,
            "problems": list(self.problems),
        }


def _review_status(sample: DatasetSample) -> str:
    if sample.review is None:
        return "unreviewed"
    return sample.review.status.value


def _matches_status(sample: DatasetSample, status: str) -> bool:
    current = _review_status(sample)
    if status == "all":
        return True
    if status == "needs-review":
        return current in {"unreviewed", "rejected"}
    return current == status


def _safe_dataset_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root not in path.parents and path != root:
        raise ValueError(f"Dataset path escapes root: {relative}")
    return path


def _make_thumbnail(
    source: Path,
    destination: Path,
    *,
    max_width: int,
    max_height: int,
) -> None:
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail(
            (max_width, max_height),
            Image.Resampling.LANCZOS,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(
            destination,
            format="JPEG",
            quality=82,
            optimize=True,
        )


def build_review_queue(
    samples: list[DatasetSample],
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    split: DatasetSplit = DatasetSplit.TEST,
    status: str = "needs-review",
    max_thumbnail_width: int = 900,
    max_thumbnail_height: int = 1200,
) -> tuple[Path, Path]:
    allowed_statuses = {
        "all",
        "needs-review",
        "unreviewed",
        DatasetReviewStatus.APPROVED.value,
        DatasetReviewStatus.REJECTED.value,
    }
    if status not in allowed_statuses:
        raise ValueError(
            "status must be one of: "
            + ", ".join(sorted(allowed_statuses))
        )
    if max_thumbnail_width < 64 or max_thumbnail_height < 64:
        raise ValueError("thumbnail dimensions must be at least 64")

    root = Path(dataset_root).resolve()
    output = Path(output_dir)
    thumbnails = output / "thumbnails"
    output.mkdir(parents=True, exist_ok=True)

    selected = sorted(
        (
            sample
            for sample in samples
            if sample.split == split and _matches_status(sample, status)
        ),
        key=lambda sample: (
            _review_status(sample),
            sample.subset.value,
            sample.id,
        ),
    )

    entries: list[ReviewQueueEntry] = []
    for sample in selected:
        problems: list[str] = []
        source = _safe_dataset_path(root, sample.source)
        truth = _safe_dataset_path(root, sample.ground_truth)

        thumbnail_relative: str | None = None
        if not source.is_file():
            problems.append(f"missing source image: {sample.source}")
        else:
            thumbnail_path = thumbnails / f"{sample.id}.jpg"
            try:
                _make_thumbnail(
                    source,
                    thumbnail_path,
                    max_width=max_thumbnail_width,
                    max_height=max_thumbnail_height,
                )
            except Exception as exc:
                problems.append(f"could not render thumbnail: {exc}")
            else:
                thumbnail_relative = thumbnail_path.relative_to(output).as_posix()

        truth_text: str | None = None
        if not truth.is_file():
            problems.append(f"missing ground truth: {sample.ground_truth}")
        else:
            try:
                truth_text = truth.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                problems.append("ground truth is not valid UTF-8")

        entries.append(
            ReviewQueueEntry(
                id=sample.id,
                document_id=sample.document_id,
                split=sample.split.value,
                subset=sample.subset.value,
                tags=tuple(sample.tags),
                source=sample.source,
                ground_truth=sample.ground_truth,
                review_status=_review_status(sample),
                reviewer=(
                    sample.review.reviewer
                    if sample.review is not None
                    else None
                ),
                review_notes=(
                    sample.review.notes
                    if sample.review is not None
                    else None
                ),
                thumbnail=thumbnail_relative,
                ground_truth_text=truth_text,
                problems=tuple(problems),
            )
        )

    payload = {
        "schema_version": "1",
        "split": split.value,
        "status_filter": status,
        "sample_count": len(entries),
        "entries": [entry.to_dict() for entry in entries],
    }
    json_path = output / "review-queue.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    html_path = output / "index.html"
    html_path.write_text(
        _render_html(payload),
        encoding="utf-8",
    )
    return html_path, json_path


def _render_html(payload: dict) -> str:
    cards: list[str] = []
    for entry in payload["entries"]:
        tags = " ".join(
            f'<span class="tag">{html.escape(tag)}</span>'
            for tag in entry["tags"]
        )
        problems = "".join(
            f"<li>{html.escape(problem)}</li>"
            for problem in entry["problems"]
        )
        problems_html = (
            f'<div class="problems"><strong>Problems</strong><ul>{problems}</ul></div>'
            if problems
            else ""
        )
        if entry["thumbnail"]:
            image_html = (
                f'<img loading="lazy" src="{html.escape(entry["thumbnail"], quote=True)}" '
                f'alt="Capture {html.escape(entry["id"], quote=True)}">'
            )
        else:
            image_html = '<div class="missing">No preview available</div>'

        truth = html.escape(entry["ground_truth_text"] or "")
        notes = html.escape(entry["review_notes"] or "")
        reviewer = html.escape(entry["reviewer"] or "")
        sample_id = html.escape(entry["id"])
        cards.append(
            f"""
<section class="card">
  <div class="preview">{image_html}</div>
  <div class="details">
    <h2>{sample_id}</h2>
    <dl>
      <dt>Status</dt><dd>{html.escape(entry["review_status"])}</dd>
      <dt>Subset</dt><dd>{html.escape(entry["subset"])}</dd>
      <dt>Document</dt><dd>{html.escape(entry["document_id"])}</dd>
      <dt>Source</dt><dd><code>{html.escape(entry["source"])}</code></dd>
      <dt>Ground truth</dt><dd><code>{html.escape(entry["ground_truth"])}</code></dd>
      <dt>Reviewer</dt><dd>{reviewer or "—"}</dd>
    </dl>
    <div class="tags">{tags}</div>
    {problems_html}
    <h3>Ground truth</h3>
    <pre>{truth}</pre>
    <h3>Existing review notes</h3>
    <p>{notes or "—"}</p>
    <h3>Review command</h3>
    <pre>lao-ocr review-dataset-sample --manifest &lt;manifest.jsonl&gt; --dataset-root &lt;dataset-root&gt; --id {sample_id} --status approved --reviewer &lt;alias&gt;</pre>
  </div>
</section>
"""
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lao OCR benchmark review queue</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 0; background: #f5f6f8; color: #16181d; }}
main {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
header {{ margin-bottom: 24px; }}
.card {{ display: grid; grid-template-columns: minmax(280px, 42%) 1fr; gap: 24px; background: white; border: 1px solid #dfe3ea; border-radius: 12px; padding: 18px; margin-bottom: 20px; }}
.preview img {{ display: block; width: 100%; height: auto; border: 1px solid #dfe3ea; }}
.missing {{ padding: 64px 16px; text-align: center; background: #f1f2f4; }}
dl {{ display: grid; grid-template-columns: 120px 1fr; gap: 6px 12px; }}
dt {{ font-weight: 700; }}
dd {{ margin: 0; overflow-wrap: anywhere; }}
.tag {{ display: inline-block; padding: 3px 7px; margin: 2px; background: #eef0ff; border-radius: 999px; font-size: 12px; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f7f8fa; border-radius: 8px; padding: 12px; }}
.problems {{ border-left: 4px solid #b42318; padding-left: 12px; color: #7a271a; }}
@media (max-width: 760px) {{ .card {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<main>
<header>
<h1>Lao OCR benchmark review queue</h1>
<p>Split: {html.escape(payload["split"])} · filter: {html.escape(payload["status_filter"])} · samples: {payload["sample_count"]}</p>
<p>Review the capture, exact transcription, tags/subset, page identity, and release metadata before recording approval.</p>
</header>
{"".join(cards) if cards else "<p>No samples match this review queue.</p>"}
</main>
</body>
</html>
"""


def review_queue_summary(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    return {
        "sample_count": len(entries),
        "with_problems": sum(
            bool(entry.get("problems"))
            for entry in entries
            if isinstance(entry, dict)
        ),
    }
