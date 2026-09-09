"""Fast digital PDF extraction using PyMuPDF (fitz).

Assigned Engineer: Engineer 1 (User)
"""

from pathlib import Path
from typing import List, Union

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from cv_segment.schemas import PagePayload, TextSpan


class PDFStreamParser:
    """High-throughput native PDF extractor converting documents into PagePayload sequences."""

    def __init__(self, min_char_threshold: int = 80) -> None:
        self.min_char_threshold = min_char_threshold

    def parse_pdf(self, pdf_path: Union[str, Path]) -> list[PagePayload]:
        """Extracts text, bounding boxes, font attributes, and images for every page in stream."""
        path_obj = Path(pdf_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if fitz is None:
            raise RuntimeError(
                "PyMuPDF is not installed. Install via `pip install PyMuPDF`."
            )

        doc = fitz.open(str(path_obj))
        pages: list[PagePayload] = []

        for idx, page in enumerate(doc):
            rect = page.rect
            text = page.get_text("text") or ""
            char_count = len(text.strip())

            # Extract detailed block and span typography
            spans: list[TextSpan] = []
            text_blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_SEARCH).get("blocks", [])

            for block in text_blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            span_text = span.get("text", "").strip()
                            if span_text:
                                bbox = span.get("bbox", (0.0, 0.0, 0.0, 0.0))
                                spans.append(
                                    TextSpan(
                                        text=span_text,
                                        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                                        font_name=span.get("font", "unknown"),
                                        font_size=float(span.get("size", 10.0)),
                                        flags=int(span.get("flags", 0)),
                                    )
                                )

            # Image detection
            image_list = page.get_images(full=True)
            has_images = len(image_list) > 0

            # Determine if page is scanned or digital
            is_scanned = char_count < self.min_char_threshold and has_images

            pages.append(
                PagePayload(
                    page_index=idx,
                    width=float(rect.width),
                    height=float(rect.height),
                    text=text,
                    char_count=char_count,
                    spans=spans,
                    has_images=has_images,
                    is_scanned=is_scanned,
                )
            )

        doc.close()
        return pages
