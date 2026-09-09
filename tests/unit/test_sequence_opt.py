"""Unit tests for Viterbi HMM Trellis Optimization."""

from cv_segment.models.sequence_opt import ViterbiSequenceOptimizer


def test_viterbi_single_page_cvs() -> None:
    optimizer = ViterbiSequenceOptimizer()
    # 3 single-page CVs -> transitions 0->1 and 1->2 are high split probabilities
    probs = [0.99, 0.99]
    boundaries = optimizer.decode(probs)
    # Expected cut points: page 1 and page 2
    assert boundaries == [1, 2]


def test_viterbi_two_page_cv() -> None:
    optimizer = ViterbiSequenceOptimizer()
    # Candidate 1: Pages 1 & 2 (prob[0] = 0.01)
    # Candidate 2: Page 3 (prob[1] = 0.98)
    probs = [0.01, 0.98]
    boundaries = optimizer.decode(probs)
    # Expected cut point: page 2
    assert boundaries == [2]


def test_viterbi_empty_input() -> None:
    optimizer = ViterbiSequenceOptimizer()
    assert optimizer.decode([]) == []
