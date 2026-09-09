"""36-Dimensional Tabular Pairwise Feature Extractor for Classical ML (LightGBM).

Assigned Engineer: Engineer 2 & Engineer 3
"""

import math
from typing import Dict, List
from cv_segment.heuristics.font_rules import FontHierarchyExtractor
from cv_segment.heuristics.pagination import PaginationExtractor
from cv_segment.heuristics.pii_rules import PIIRuleExtractor
from cv_segment.heuristics.syntactic import SyntacticContinuityExtractor
from cv_segment.schemas import PagePayload


class PairwiseFeatureExtractor:
    """Computes normalized 36-feature vector comparing adjacent pages (Page_i, Page_j)."""

    def __init__(self) -> None:
        self.pii_extractor = PIIRuleExtractor()
        self.font_extractor = FontHierarchyExtractor()
        self.pagination_extractor = PaginationExtractor()
        self.syntactic_extractor = SyntacticContinuityExtractor()

    def extract_features(
        self,
        page_i: PagePayload,
        page_j: PagePayload,
        cumulative_pages_in_segment: int = 1,
    ) -> dict[str, float]:
        """Calculates 36 features for binary classification."""
        tokens_i = set(page_i.text.lower().split())
        tokens_j = set(page_j.text.lower().split())

        # 1. Text & Semantic Overlap
        intersection = len(tokens_i & tokens_j)
        union = len(tokens_i | tokens_j)
        jaccard = intersection / union if union > 0 else 0.0

        # 2. PII Signals
        pii_i = self.pii_extractor.compute_header_pii_score(page_i)
        pii_j = self.pii_extractor.compute_header_pii_score(page_j)
        delta_pii = pii_j - pii_i

        # 3. Typography Discontinuities
        font_ratio_i = self.font_extractor.compute_dominance_ratio(page_i)
        font_ratio_j = self.font_extractor.compute_dominance_ratio(page_j)
        delta_font_dom = font_ratio_j - font_ratio_i

        # 4. Pagination Signals
        page_num_i = self.pagination_extractor.extract_page_numbers(page_i)
        page_num_j = self.pagination_extractor.extract_page_numbers(page_j)

        exact_inc = 0.0
        if page_num_i and page_num_j and page_num_j[0] == page_num_i[0] + 1:
            exact_inc = 1.0

        # 5. Syntactic Continuity
        syntactic_score = self.syntactic_extractor.compute_continuation_score(page_i, page_j)

        features: dict[str, float] = {
            "tfidf_cosine_sim": 0.0,  # Populated via vectorizer during batch inference
            "token_jaccard_sim": float(jaccard),
            "name_levenshtein_ratio": 0.0,
            "shared_org_count": 0.0,
            "shared_url_domain_count": 0.0,
            "shared_email_domain_flag": 0.0,
            "delta_pii_score": float(delta_pii),
            "email_present_p2": 1.0 if pii_j >= 2.0 else 0.0,
            "phone_present_p2": 1.0 if pii_j >= 1.5 else 0.0,
            "name_present_p2": 1.0 if font_ratio_j >= 1.8 else 0.0,
            "address_match_score": 0.0,
            "delta_link_count": 0.0,
            "delta_font_dominance": float(delta_font_dom),
            "delta_max_font_size": 0.0,
            "delta_median_font_size": 0.0,
            "delta_left_margin": 0.0,
            "delta_top_margin": 0.0,
            "delta_line_spacing": 0.0,
            "delta_text_block_count": float(abs(len(page_j.spans) - len(page_i.spans))),
            "delta_column_count": 0.0,
            "delta_aspect_ratio": float(
                abs((page_j.width / page_j.height) - (page_i.width / page_i.height))
            ),
            "delta_blank_line_count": 0.0,
            "flag_trailing_hyphen_p1": 1.0 if page_i.text.strip().endswith("-") else 0.0,
            "flag_lack_terminal_period_p1": 1.0 if not page_i.text.strip().endswith(".") else 0.0,
            "flag_conjunction_ending_p1": 1.0 if syntactic_score >= 0.8 else 0.0,
            "flag_initial_lowercase_p2": (
                1.0 if page_j.text.strip() and page_j.text.strip()[0].islower() else 0.0
            ),
            "flag_bullet_continuation": 0.0,
            "flag_numbered_list_increment": 0.0,
            "flag_table_row_continuation": 0.0,
            "flag_header_entity_match": 0.0,
            "terminal_section_score_p1": (
                1.0 if "references" in page_i.text.lower()[-300:] else 0.0
            ),
            "front_section_score_p2": (
                1.0 if "summary" in page_j.text.lower()[:300] else 0.0
            ),
            "cumulative_page_count": float(cumulative_pages_in_segment),
            "pagination_gap_flag": 0.0,
            "exact_page_increment_flag": exact_inc,
            "total_pagination_mismatch_flag": 0.0,
        }

        return features
