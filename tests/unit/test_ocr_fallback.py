"""Unit tests for conditional OCR fallback."""

from cv_segment.ingestion.ocr_fallback import ConditionalOCRFallback
from cv_segment.schemas import PagePayload


def test_ocr_fallback_skips_digital(sample_page_start: PagePayload) -> None:
    ocr = ConditionalOCRFallback()
    page_out = ocr.process_if_needed(sample_page_start)
    assert not page_out.is_scanned
    assert page_out.text == sample_page_start.text


def test_ocr_fallback_processes_scanned() -> None:
    ocr = ConditionalOCRFallback(char_threshold=50)
    scanned_page = PagePayload(
        page_index=0,
        width=612.0,
        height=792.0,
        text="",
        char_count=0,
        has_images=True,
        is_scanned=True,
    )
    page_out = ocr.process_if_needed(scanned_page)
    assert page_out.is_scanned
