from services.api.app.metrics import ApiMetrics, normalize_route


def test_job_routes_are_normalized() -> None:
    assert normalize_route("/v1/jobs/batch") == "/v1/jobs/batch"
    assert normalize_route("/v1/jobs/abc123") == "/v1/jobs/{job_id}"
    assert (
        normalize_route("/v1/jobs/abc123/download")
        == "/v1/jobs/{job_id}/download"
    )
    assert normalize_route("/health") == "/health"


def test_metrics_render_http_and_job_series() -> None:
    metrics = ApiMetrics()
    metrics.observe_request(
        method="get",
        path="/v1/jobs/abc123",
        status_code=200,
        duration_seconds=0.25,
    )
    metrics.observe_request(
        method="GET",
        path="/v1/jobs/def456",
        status_code=404,
        duration_seconds=0.10,
    )

    rendered = metrics.render_prometheus(
        {
            "current_by_status": {"running": 2},
            "completed_total": {"succeeded": 4, "failed": 1},
            "duration_seconds_sum": {"succeeded": 8.5, "failed": 0.4},
        }
    )

    assert 'route="/v1/jobs/{job_id}"' in rendered
    assert "abc123" not in rendered
    assert "def456" not in rendered
    assert (
        'lao_ocr_http_requests_total{method="GET",'
        'route="/v1/jobs/{job_id}",status="200"} 1'
        in rendered
    )
    assert 'lao_ocr_jobs_current{status="running"} 2' in rendered
    assert 'lao_ocr_jobs_completed_total{status="succeeded"} 4' in rendered
    assert 'lao_ocr_job_duration_seconds_sum{status="succeeded"} 8.500000000' in rendered
