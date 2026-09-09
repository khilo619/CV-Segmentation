"""Unit tests for Tier 1 Heuristics rules."""

from cv_segment.heuristics.engine import HeuristicGatingEngine
from cv_segment.heuristics.font_rules import FontHierarchyExtractor
from cv_segment.heuristics.pagination import PaginationExtractor
from cv_segment.heuristics.pii_rules import PIIRuleExtractor
from cv_segment.heuristics.syntactic import SyntacticContinuityExtractor
from cv_segment.schemas import PagePayload


def test_pii_rule_extractor(sample_page_start: PagePayload, sample_page_continuation: PagePayload) -> None:
    extractor = PIIRuleExtractor()
    score_start = extractor.compute_header_pii_score(sample_page_start)
    score_cont = extractor.compute_header_pii_score(sample_page_continuation)

    assert score_start >= 3.0  # Has email, phone, linkedin
    assert score_cont == 0.0   # No PII in header


def test_font_hierarchy_extractor(sample_page_start: PagePayload, sample_page_continuation: PagePayload) -> None:
    extractor = FontHierarchyExtractor()
    ratio_start = extractor.compute_dominance_ratio(sample_page_start)
    ratio_cont = extractor.compute_dominance_ratio(sample_page_continuation)

    assert ratio_start >= 1.8  # 24pt title vs 10pt median
    assert ratio_cont < 1.5


def test_syntactic_continuity() -> None:
    extractor = SyntacticContinuityExtractor()
    page1 = PagePayload(
        page_index=0, width=600, height=800, text="Responsibilities include managing teams and", char_count=45
    )
    page2 = PagePayload(
        page_index=1, width=600, height=800, text="overseeing production deployments.", char_count=35
    )
    score = extractor.compute_continuation_score(page1, page2)
    assert score > 0.7  # Dangling conjunction + lowercase continuation


def test_heuristic_gating_engine(sample_page_start: PagePayload, sample_page_continuation: PagePayload) -> None:
    engine = HeuristicGatingEngine()
    # Transition from continuation page back to a new candidate start page
    eval_split = engine.evaluate_pair(sample_page_continuation, sample_page_start)
    assert eval_split.overall_confidence >= 0.90
