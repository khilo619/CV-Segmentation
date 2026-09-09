"""Typography and Structural Hierarchy Analyzer.

Assigned Engineer: Engineer 2
"""

import statistics

from cv_segment.schemas import PagePayload


class FontHierarchyExtractor:
    """Detects candidate name titles and typography discontinuities."""

    def __init__(self, header_ratio: float = 0.25) -> None:
        self.header_ratio = header_ratio

    def compute_dominance_ratio(self, page: PagePayload) -> float:
        """Calculates ratio of max font in top 25% vs page median font size."""
        if not page.spans:
            return 1.0

        all_font_sizes = [s.font_size for s in page.spans]
        median_size = statistics.median(all_font_sizes) if all_font_sizes else 10.0
        if median_size <= 0:
            median_size = 10.0

        header_y_max = page.height * self.header_ratio
        header_fonts = [s.font_size for s in page.spans if s.bbox[1] <= header_y_max]

        max_header_font = max(header_fonts) if header_fonts else median_size
        return float(max_header_font / median_size)
