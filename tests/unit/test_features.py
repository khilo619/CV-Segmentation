"""Unit tests for Tier 2 Pairwise Feature Vector Extractor."""

from cv_segment.features.pairwise import PairwiseFeatureExtractor
from cv_segment.schemas import PagePayload


def test_pairwise_feature_extraction(
    sample_page_start: PagePayload, sample_page_continuation: PagePayload
) -> None:
    extractor = PairwiseFeatureExtractor()
    features = extractor.extract_features(sample_page_start, sample_page_continuation)

    # Verify 36 features exist
    assert len(features) == 36
    assert "token_jaccard_sim" in features
    assert "delta_pii_score" in features
    assert "delta_font_dominance" in features
    assert "cumulative_page_count" in features

    # Values check
    assert isinstance(features["token_jaccard_sim"], float)
    assert features["token_jaccard_sim"] >= 0.0
