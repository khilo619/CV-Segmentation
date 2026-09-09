"""Cross-Page Syntactic & Sentence Fracture Analyzer.

Assigned Engineer: Engineer 2
"""

from cv_segment.schemas import PagePayload


class SyntacticContinuityExtractor:
    """Detects trailing hyphens, dangling conjunctions, and sentence completions."""

    TERMINAL_PUNCTUATION = {".", "!", "?", ":"}
    CONJUNCTIONS = {"and", "or", "with", "including", "such as", "to", "for", "in", "of"}

    def compute_continuation_score(self, page_i: PagePayload, page_j: PagePayload) -> float:
        """Evaluates whether page_j syntactically continues sentence from page_i."""
        text_i = page_i.text.strip()
        text_j = page_j.text.strip()

        if not text_i or not text_j:
            return 0.0

        last_line_i = text_i.splitlines()[-1].strip() if text_i.splitlines() else ""
        first_line_j = text_j.splitlines()[0].strip() if text_j.splitlines() else ""

        score = 0.0

        # Trailing hyphen check (e.g. "responsibili-" -> "ties include")
        if last_line_i.endswith("-") and first_line_j and first_line_j[0].islower():
            score += 0.95

        # Dangling non-terminal sentence ending
        if last_line_i and last_line_i[-1] not in self.TERMINAL_PUNCTUATION:
            words = last_line_i.split()
            if words and words[-1].lower() in self.CONJUNCTIONS:
                score += 0.85
            elif first_line_j and first_line_j[0].islower():
                score += 0.75

        return min(1.0, score)
