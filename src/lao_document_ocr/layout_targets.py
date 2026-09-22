from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from lao_document_ocr.layout_ground_truth import (
    load_layout_ground_truth,
)
from lao_document_ocr.models import BlockType, Document

LAYOUT_CLASS_IDS = {
    "background": 0,
    BlockType.HEADING.value: 1,
    BlockType.PARAGRAPH.value: 2,
    BlockType.LIST.value: 3,
    BlockType.TABLE.value: 4,
    BlockType.IMAGE.value: 5,
}


class LayoutTargetError(ValueError):
    pass


@dataclass(frozen=True)
class LayoutBoxTarget:
    class_id: int
    class_name: str
    x: int
    y: int
    width: int
    height: int
    order: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LayoutTargetArtifact:
    id: str
    image: str
    layout_ground_truth: str
    mask: str
    boxes: str
    split: str
    subset: str
    tags: tuple[str, ...]
    width: int
    height: int

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["tags"] = list(self.tags)
        return payload


def _single_page(document: Document):
    if len(document.pages) != 1:
        raise LayoutTargetError(
            "Layout targets require exactly one annotated page"
        )
    return document.pages[0]


def render_layout_class_mask(
    document: Document,
) -> Image.Image:
    page = _single_page(document)
    mask = np.zeros(
        (page.height, page.width),
        dtype=np.uint8,
    )

    for block_index, block in enumerate(page.blocks):
        if block.bbox is None:
            raise LayoutTargetError(
                f"block {block_index}: bbox is required"
            )
        class_id = LAYOUT_CLASS_IDS[block.type.value]
        box = block.bbox
        region = mask[
            box.y : box.y + box.height,
            box.x : box.x + box.width,
        ]
        conflicting = (region != 0) & (region != class_id)
        if np.any(conflicting):
            raise LayoutTargetError(
                f"block {block_index}: overlaps another semantic class"
            )
        region[:] = class_id

    return Image.fromarray(mask)


def build_layout_box_targets(
    document: Document,
) -> list[LayoutBoxTarget]:
    page = _single_page(document)
    targets: list[LayoutBoxTarget] = []

    for order, block in enumerate(page.blocks):
        if block.bbox is None:
            raise LayoutTargetError(
                f"block {order}: bbox is required"
            )
        box = block.bbox
        targets.append(
            LayoutBoxTarget(
                class_id=LAYOUT_CLASS_IDS[block.type.value],
                class_name=block.type.value,
                x=box.x,
                y=box.y,
                width=box.width,
                height=box.height,
                order=order,
            )
        )

    return targets


def write_layout_target_artifacts(
    *,
    sample_id: str,
    image_path: str,
    layout_ground_truth_path: str,
    split: str,
    subset: str,
    tags: list[str] | tuple[str, ...],
    dataset_root: str | Path,
    output_dir: str | Path,
) -> LayoutTargetArtifact:
    root = Path(dataset_root)
    output = Path(output_dir)
    image_file = root / image_path
    layout_path = root / layout_ground_truth_path
    if not image_file.is_file():
        raise FileNotFoundError(f"Layout training image not found: {image_file}")
    document = load_layout_ground_truth(layout_path)
    page = _single_page(document)
    try:
        with Image.open(image_file) as source_image:
            image_size = source_image.size
    except Exception as exc:
        raise LayoutTargetError(
            f"Could not open layout training image: {image_file}"
        ) from exc
    if image_size != (page.width, page.height):
        raise LayoutTargetError(
            "Layout target image dimensions do not match annotation "
            f"({image_size[0]}x{image_size[1]} != "
            f"{page.width}x{page.height})"
        )

    mask = render_layout_class_mask(document)
    boxes = build_layout_box_targets(document)

    mask_relative = Path("masks") / f"{sample_id}.png"
    boxes_relative = Path("boxes") / f"{sample_id}.json"
    mask_path = output / mask_relative
    boxes_path = output / boxes_relative
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    boxes_path.parent.mkdir(parents=True, exist_ok=True)

    mask.save(mask_path, format="PNG", optimize=True)
    boxes_path.write_text(
        json.dumps(
            [target.to_dict() for target in boxes],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return LayoutTargetArtifact(
        id=sample_id,
        image=image_path,
        layout_ground_truth=layout_ground_truth_path,
        mask=mask_relative.as_posix(),
        boxes=boxes_relative.as_posix(),
        split=split,
        subset=subset,
        tags=tuple(tags),
        width=page.width,
        height=page.height,
    )


def write_layout_target_manifest(
    artifacts: list[LayoutTargetArtifact],
    output_dir: str | Path,
) -> Path:
    if not artifacts:
        raise LayoutTargetError(
            "Cannot write an empty layout target manifest"
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    classes_path = output / "classes.json"
    classes_path.write_text(
        json.dumps(
            LAYOUT_CLASS_IDS,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = output / "targets.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for artifact in sorted(
            artifacts,
            key=lambda item: item.id,
        ):
            handle.write(
                json.dumps(
                    artifact.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return manifest_path
