"""Explicit Header/Footer Pagination Sequence Analyzer.

Assigned Engineer: Engineer 2
"""

import re
from typing import Optional, Tuple
from cv_segment.schemas import PagePayload


PAGE_X_OF_Y_REGEX = re.compile(r"(?i)page\s*(\d+)\s*(?:of|/)\s*(\d+)")
SOLITARY_NUM_REGEX = re.compile(r"(?i)^\s*[-–—]?\s*(\d+)\s*[-–—]?\s*$")


class PaginationExtractor:
    """Parses explicit pagination patterns in top/bottom 10% margins."""

    def __init__(self, margin_ratio: float = 0.10) -> None:
        self.margin_ratio = margin_ratio

    def extract_page_numbers(self, page: PagePayload) -> Optional[tuple[int, Optional[int]]]:
        """Returns tuple of (current_page, total_pages) if detected in margins."""
        header_limit = page.height * self.margin_ratio
        footer_limit = page.height * (1.0 - self.margin_ratio)

        margin_spans = [
            s.text
            for s in page.spans
            if s.bbox[1] <= header_limit or s.bbox[3] >= footer_limit
        ]
        margin_text = " \n ".join(margin_spans)

        # Check Page X of Y
        match_xy = PAGE_X_OF_Y_REGEX.search(margin_text)
        if match_xy:
            try:
                curr = int(match_xy.group(1))
                total = int(match_xy.group(2))
                return curr, total
            except ValueError:
                pass

        # Check solitary number lines in margins
        for line in margin_text.splitlines():
            line_clean = line.strip()
            match_num = SOLITARY_NUM_REGEX.match(line_clean)
            if match_num:
                try:
                    curr = int(match_num.group(1))
                    return curr, None
                except ValueError:
                    pass

        return None
