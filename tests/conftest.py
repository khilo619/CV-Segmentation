"""Shared test fixtures for unit and integration testing."""

import pytest

from cv_segment.schemas import PagePayload, TextSpan


@pytest.fixture
def sample_page_start() -> PagePayload:
    """Simulates Page 1 of a resume with candidate name, contact info, and title."""
    spans = [
        TextSpan(
            text="JOHN DOE",
            bbox=(50.0, 50.0, 200.0, 80.0),
            font_name="Helvetica-Bold",
            font_size=24.0,
        ),
        TextSpan(
            text="john.doe@example.com | +1 (555) 019-2834 | linkedin.com/in/johndoe",
            bbox=(50.0, 85.0, 450.0, 100.0),
            font_name="Helvetica",
            font_size=10.0,
        ),
        TextSpan(
            text="Senior Software Engineer with 8 years of experience in distributed systems.",
            bbox=(50.0, 120.0, 500.0, 140.0),
            font_name="Helvetica",
            font_size=11.0,
        ),
    ]
    return PagePayload(
        page_index=0,
        width=612.0,
        height=792.0,
        text="JOHN DOE\njohn.doe@example.com | +1 (555) 019-2834 | linkedin.com/in/johndoe\nSenior Software Engineer",
        char_count=120,
        spans=spans,
        has_images=False,
        is_scanned=False,
    )


@pytest.fixture
def sample_page_continuation() -> PagePayload:
    """Simulates Page 2 of a resume (continuation with regular body text)."""
    spans = [
        TextSpan(
            text="Work Experience (Continued)",
            bbox=(50.0, 50.0, 250.0, 70.0),
            font_name="Helvetica-Bold",
            font_size=14.0,
        ),
        TextSpan(
            text="Led team of 6 engineers to migrate legacy backend to microservices.",
            bbox=(50.0, 80.0, 500.0, 100.0),
            font_name="Helvetica",
            font_size=10.0,
        ),
    ]
    return PagePayload(
        page_index=1,
        width=612.0,
        height=792.0,
        text="Work Experience (Continued)\nLed team of 6 engineers to migrate legacy backend to microservices.",
        char_count=95,
        spans=spans,
        has_images=False,
        is_scanned=False,
    )
