"""Tier 1 Heuristic Gating Engine: Evaluates deterministic rules for adjacent pages.

Assigned Engineer: Engineer 2
"""

from typing import List
from cv_segment.schemas import HeuristicEvaluation, PagePayload
from .font_rules import FontHierarchyExtractor
from .pagination import PaginationExtractor
from .pii_rules import PIIRuleExtractor
from .syntactic import SyntacticContinuityExtractor


class HeuristicGatingEngine:
    """Orchestrates heuristic rule evaluation and short-circuits obvious page pairs."""

    def __init__(
        self,
        split_threshold: float = 0.98,
        cont_threshold: float = 0.02,
    ) -> None:
        self.split_threshold = split_threshold
        self.cont_threshold = cont_threshold
        self.pii_extractor = PIIRuleExtractor()
        self.font_extractor = FontHierarchyExtractor()
        self.pagination_extractor = PaginationExtractor()
        self.syntactic_extractor = SyntacticContinuityExtractor()

    def evaluate_pair(self, page_i: PagePayload, page_j: PagePayload) -> HeuristicEvaluation:
        """Computes rule signals and returns boundary confidence score."""
        pii_score_j = self.pii_extractor.compute_header_pii_score(page_j)
        font_ratio_j = self.font_extractor.compute_dominance_ratio(page_j)
        syntactic_score = self.syntactic_extractor.compute_continuation_score(page_i, page_j)

        # Pagination logic
        page_num_i = self.pagination_extractor.extract_page_numbers(page_i)
        page_num_j = self.pagination_extractor.extract_page_numbers(page_j)

        is_explicit_boundary = False
        is_explicit_continuation = False

        if page_num_j and page_num_j[0] == 1:
            is_explicit_boundary = True
        elif page_num_i and page_num_j:
            curr_i = page_num_i[0]
            curr_j = page_num_j[0]
            total_i = page_num_i[1]
            if curr_j == curr_i + 1:
                is_explicit_continuation = True
            elif total_i and curr_i == total_i:
                is_explicit_boundary = True

        # Compute preliminary confidence score
        if is_explicit_boundary:
            confidence = 1.0
        elif is_explicit_continuation:
            confidence = 0.0
        elif syntactic_score > 0.8:
            confidence = 0.01
        elif pii_score_j >= 3.0 and font_ratio_j >= 1.5:
            confidence = 0.99
        else:
            # Ambiguous band, interpolate soft score
            norm_pii = min(1.0, pii_score_j / 4.0)
            norm_font = min(1.0, max(0.0, (font_ratio_j - 1.0) / 2.0))
            confidence = 0.5 * norm_pii + 0.3 * norm_font + 0.2 * (1.0 - syntactic_score)

        short_circuit = (confidence >= self.split_threshold) or (confidence <= self.cont_threshold)

        return HeuristicEvaluation(
            pair_index=page_i.page_index,
            pii_score=pii_score_j,
            font_dominance_ratio=font_ratio_j,
            is_explicit_boundary=is_explicit_boundary,
            is_explicit_continuation=is_explicit_continuation,
            syntactic_continuation_score=syntactic_score,
            overall_confidence=confidence,
            short_circuit=short_circuit,
        )
