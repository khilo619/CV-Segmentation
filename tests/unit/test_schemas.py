"""Unit tests for data schemas and dataclass models."""

from cv_segment.schemas import (
    CandidateSegment,
    PagePayload,
    SegmentationManifest,
    TextSpan,
)


def test_text_span_creation() -> None:
    span = TextSpan(
        text="Sample text",
        bbox=(10.0, 20.0, 100.0, 40.0),
        font_name="Arial",
        font_size=12.0,
    )
    assert span.text == "Sample text"
    assert span.font_size == 12.0
    assert span.bbox == (10.0, 20.0, 100.0, 40.0)


def test_page_payload_defaults() -> None:
    page = PagePayload(
        page_index=0,
        width=595.0,
        height=842.0,
        text="Hello World",
        char_count=11,
    )
    assert page.page_index == 0
    assert len(page.spans) == 0
    assert not page.has_images
    assert not page.is_scanned


def test_segmentation_manifest() -> None:
    segment = CandidateSegment(
        candidate_id="candidate_001",
        start_page=1,
        end_page=2,
        page_count=2,
    )
    manifest = SegmentationManifest(
        stream_filename="batch.pdf",
        total_pages=2,
        total_candidates=1,
        boundaries=[],
        segments=[segment],
        processing_time_ms=15.4,
    )
    assert manifest.total_candidates == 1
    assert manifest.segments[0].page_count == 2
