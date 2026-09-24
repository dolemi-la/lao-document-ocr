from pathlib import Path

WORKFLOW = (
    Path(__file__).parents[1]
    / ".github"
    / "workflows"
    / "remote-evaluation.yml"
)


def test_remote_evaluation_workflow_is_manual_and_read_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in text
    assert "contents: read" in text
    assert "push:" not in text
    assert "pull_request:" not in text


def test_remote_evaluation_workflow_installs_real_tesseract_languages() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "tesseract-ocr-lao" in text
    assert "tesseract-ocr-eng" in text
    assert "tesseract --list-langs" in text
    assert "lao-ocr evaluate-remote-sources" in text
    assert "--source-id \"$SOURCE_ID\"" in text


def test_remote_evaluation_workflow_exposes_opt_in_auto_orientation() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "auto_orient_right_angles:" in text
    assert "diagnostic_rotation_probes:" in text
    assert "AUTO_ORIENT: ${{ inputs.auto_orient_right_angles }}" in text
    assert "ROTATION_PROBES: ${{ inputs.diagnostic_rotation_probes }}" in text
    assert "args+=(--auto-orient-right-angles)" in text
    assert "args+=(--rotation-probes)" in text
    assert 'payload["selection"]["auto_orient_right_angles"]' in text
    assert 'payload["selection"]["probe_right_angle_rotations"]' in text



def test_remote_evaluation_workflow_preserves_privacy_boundary() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Verify report privacy boundary" in text
    assert '"not_benchmark_accuracy"' in text
    assert '"ground_truth"' in text
    assert '"hypothesis"' in text
    assert "actions/upload-artifact@v7" in text
    assert "retention-days: 14" in text


def test_remote_evaluation_workflow_uploads_report_before_failing_source() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "id: diagnostic" in text
    assert 'echo "exit_code=$code" >> "$GITHUB_OUTPUT"' in text
    assert "name: remote-source-diagnostic-${{ github.run_id }}" in text
    assert "Fail when diagnostic source failed" in text
    upload_offset = text.index("- name: Upload diagnostic report")
    failure_offset = text.index("- name: Fail when diagnostic source failed")
    assert upload_offset < failure_offset

