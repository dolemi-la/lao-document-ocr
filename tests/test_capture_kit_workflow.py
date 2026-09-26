from pathlib import Path

WORKFLOW = (
    Path(__file__).parents[1]
    / ".github"
    / "workflows"
    / "capture-kit.yml"
)


def test_capture_kit_workflow_is_manual_and_read_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in text
    assert "contents: read" in text
    assert "push:" not in text
    assert "pull_request:" not in text


def test_capture_kit_workflow_builds_and_verifies_safe_artifact() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "PhetsarathOT-v4.103.zip" in text
    assert (
        "a97a317eb1e95c5338a38233176189f63e5b23f29a19630ec317aedee0ce8381"
        in text
    )
    assert (
        "8dd0fa55de186b051433255d80217007b4026dfc38e2781659424ad7058bbd6e"
        in text
    )
    assert "--require-complete-font" in text
    assert 'assert manifest["font"] == "PhetsarathOT-Regular.ttf"' in text
    assert "generate-capture-suite" in text
    assert "build-capture-kit" in text
    assert 'assert "ground_truth" not in collector_sheet' in text
    assert 'assert "digital_page" not in collector_sheet' in text
    assert 'assert "pack_manifest" not in collector_sheet' in text
    assert 'assert manifest["page_count"] == 60' in text
    assert "SHA256SUMS" in text


def test_capture_kit_workflow_uploads_only_collector_zip_and_checksum() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    upload_section = text.split("- name: Upload collector kit", 1)[1]

    assert "actions/upload-artifact@v7" in upload_section
    assert "project-authored-lao-v1.collector.zip" in upload_section
    assert "project-authored-lao-v1.collector.zip.sha256" in upload_section
    assert "capture-suite.json" not in upload_section
    assert "ground-truth" not in upload_section
    assert "/pages/" not in upload_section
    assert "retention-days: 30" in upload_section
