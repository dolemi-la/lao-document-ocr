import json

import numpy as np
from PIL import Image

from lao_document_ocr.layout_ground_truth import (
    write_layout_ground_truth,
)
from lao_document_ocr.layout_targets import (
    LAYOUT_CLASS_IDS,
    LayoutTargetError,
    build_layout_box_targets,
    render_layout_class_mask,
    write_layout_target_artifacts,
    write_layout_target_manifest,
)
from lao_document_ocr.models import (
    Block,
    BlockType,
    BoundingBox,
    Document,
    Page,
)


def _document() -> Document:
    return Document(
        pages=[
            Page(
                number=1,
                width=200,
                height=120,
                blocks=[
                    Block(
                        type=BlockType.HEADING,
                        text="Heading",
                        bbox=BoundingBox(
                            x=10,
                            y=10,
                            width=180,
                            height=25,
                        ),
                    ),
                    Block(
                        type=BlockType.PARAGRAPH,
                        text="Body",
                        bbox=BoundingBox(
                            x=20,
                            y=50,
                            width=160,
                            height=45,
                        ),
                    ),
                ],
            )
        ]
    )


def test_render_layout_class_mask_uses_stable_class_ids() -> None:
    mask = render_layout_class_mask(_document())
    array = np.asarray(mask)

    assert mask.size == (200, 120)
    assert array[15, 15] == LAYOUT_CLASS_IDS["heading"]
    assert array[60, 30] == LAYOUT_CLASS_IDS["paragraph"]
    assert array[110, 10] == LAYOUT_CLASS_IDS["background"]


def test_layout_box_targets_preserve_ast_order() -> None:
    targets = build_layout_box_targets(_document())

    assert [target.class_name for target in targets] == [
        "heading",
        "paragraph",
    ]
    assert [target.order for target in targets] == [0, 1]
    assert targets[0].class_id == LAYOUT_CLASS_IDS["heading"]


def test_conflicting_semantic_overlap_is_rejected() -> None:
    document = Document(
        pages=[
            Page(
                number=1,
                width=100,
                height=100,
                blocks=[
                    Block(
                        type=BlockType.HEADING,
                        bbox=BoundingBox(
                            x=10,
                            y=10,
                            width=60,
                            height=40,
                        ),
                    ),
                    Block(
                        type=BlockType.PARAGRAPH,
                        bbox=BoundingBox(
                            x=30,
                            y=20,
                            width=60,
                            height=40,
                        ),
                    ),
                ],
            )
        ]
    )

    try:
        render_layout_class_mask(document)
    except LayoutTargetError as exc:
        assert "overlaps another semantic class" in str(exc)
    else:
        raise AssertionError("Expected overlapping semantic classes to fail")


def test_write_layout_target_artifacts_and_manifest(tmp_path) -> None:
    layout = tmp_path / "layout.json"
    image = tmp_path / "data" / "clean-print" / "sample-001.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (200, 120), "white").save(image)
    write_layout_ground_truth(_document(), layout)

    artifact = write_layout_target_artifacts(
        sample_id="sample-001",
        image_path="data/clean-print/sample-001.png",
        layout_ground_truth_path=layout.name,
        split="train",
        subset="clean-print",
        tags=["layout:plain"],
        dataset_root=tmp_path,
        output_dir=tmp_path / "targets",
    )
    manifest = write_layout_target_manifest(
        [artifact],
        tmp_path / "targets",
    )

    assert (tmp_path / "targets" / artifact.mask).is_file()
    assert (tmp_path / "targets" / artifact.boxes).is_file()

    classes = json.loads(
        (tmp_path / "targets" / "classes.json").read_text(
            encoding="utf-8"
        )
    )
    assert classes["background"] == 0
    assert classes["table"] == 4

    payload = json.loads(
        manifest.read_text(encoding="utf-8")
    )
    assert payload["id"] == "sample-001"
    assert payload["mask"] == "masks/sample-001.png"
    assert payload["boxes"] == "boxes/sample-001.json"


def test_layout_target_rejects_image_dimension_mismatch(tmp_path) -> None:
    layout = tmp_path / "layout-mismatch.json"
    image = tmp_path / "mismatch.png"
    Image.new("RGB", (201, 120), "white").save(image)
    write_layout_ground_truth(_document(), layout)

    try:
        write_layout_target_artifacts(
            sample_id="mismatch",
            image_path=image.name,
            layout_ground_truth_path=layout.name,
            split="train",
            subset="clean-print",
            tags=[],
            dataset_root=tmp_path,
            output_dir=tmp_path / "targets",
        )
    except LayoutTargetError as exc:
        assert "do not match annotation" in str(exc)
    else:
        raise AssertionError("Expected image/layout dimension mismatch to fail")
