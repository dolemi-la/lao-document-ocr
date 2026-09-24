from pathlib import Path

WORKFLOW = (
    Path(__file__).parents[1]
    / ".github"
    / "workflows"
    / "remote-evaluation-suite.yml"
)


def test_remote_suite_workflow_is_manual_and_read_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in text
    assert "auto_orient_right_angles:" in text
    assert "type: boolean" in text
    assert "default: false" in text
    assert "contents: read" in text
    assert "push:" not in text
    assert "pull_request:" not in text


def test_remote_suite_workflow_uses_curated_manifest_and_tesseract() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "benchmarks/remote-diagnostic-suite.json" in text
    assert "validate_suite_against_registry" in text
    assert "tesseract-ocr-lao" in text
    assert "tesseract-ocr-eng" in text
    assert "lao-ocr evaluate-remote-suite" in text
    assert "args+=(--auto-orient-right-angles)" in text
    assert "AUTO_ORIENT:" in text
    assert "inputs.auto_orient_right_angles" in text


def test_remote_suite_workflow_preserves_privacy_and_partial_report() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert '"remote-source-diagnostic-suite"' in text
    assert '"not_benchmark_accuracy"' in text
    assert 'payload["selection"]["auto_orient_right_angles"]' in text
    assert '"ground_truth"' in text
    assert '"hypothesis"' in text
    assert "actions/upload-artifact@v7" in text
    assert "retention-days: 14" in text
    assert 'echo "exit_code=$code" >> "$GITHUB_OUTPUT"' in text
    upload_offset = text.index("- name: Upload suite diagnostic report")
    fail_offset = text.index("- name: Fail when suite has source errors")
    assert upload_offset < fail_offset
