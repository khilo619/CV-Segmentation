"""Conditional OCR fallback integration for scanned or flatbed image resume pages.

Assigned Engineer: Engineer 1 (User)
"""

from cv_segment.schemas import PagePayload


class ConditionalOCRFallback:
    """Triage engine that runs OCR strictly on pages with char_count < threshold."""

    def __init__(self, char_threshold: int = 80) -> None:
        self.char_threshold = char_threshold

    def process_if_needed(self, page: PagePayload) -> PagePayload:
        """If page is flagged as scanned, populates synthetic text spans via OCR."""
        if not page.is_scanned:
            return page

        # Placeholder: When paddleocr or tesseract is activated,
        # rasterize page and populate text spans.
        return page
