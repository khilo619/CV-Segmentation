"""Unit tests for FastAPI endpoints."""

from cv_segment.api.app import app
from cv_segment.api.routes import health_check


def test_health_check() -> None:
    res = health_check()
    assert res["status"] == "healthy"
    assert res["service"] == "cv-stream-segmenter"
    assert res["version"] == "1.0.0"


def test_app_instance() -> None:
    assert app.title == "CV Stream Segmentation API"
