"""PII and Contact Information Density Extractor.

Assigned Engineer: Engineer 2
"""

import re

from cv_segment.schemas import PagePayload

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
LINKEDIN_REGEX = re.compile(r"(?:linkedin\.com/in/|linkedin\.com/pub/)[a-zA-Z0-9_-]+")
GITHUB_REGEX = re.compile(r"github\.com/[a-zA-Z0-9_-]+")
PHONE_REGEX = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}")


class PIIRuleExtractor:
    """Evaluates density of candidate contact credentials in header zones."""

    def __init__(self, header_ratio: float = 0.35) -> None:
        self.header_ratio = header_ratio

    def compute_header_pii_score(self, page: PagePayload) -> float:
        """Calculates weighted PII presence in the top 35% vertical zone."""
        header_y_max = page.height * self.header_ratio

        # Collect text exclusively from spans located in header region
        header_spans = [s.text for s in page.spans if s.bbox[1] <= header_y_max]
        header_text = " ".join(header_spans) if header_spans else page.text[:500]

        has_email = 1.0 if EMAIL_REGEX.search(header_text) else 0.0
        has_phone = 1.0 if PHONE_REGEX.search(header_text) else 0.0
        has_linkedin = 1.0 if LINKEDIN_REGEX.search(header_text) else 0.0
        has_github = 1.0 if GITHUB_REGEX.search(header_text) else 0.0

        # Weighted PII score
        score = 2.0 * has_email + 1.5 * has_phone + 1.0 * has_linkedin + 1.0 * has_github
        return score
