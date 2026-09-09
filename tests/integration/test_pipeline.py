"""Integration tests for the End-to-End Segmentation Pipeline."""

from unittest.mock import MagicMock, patch

from cv_segment.pipeline import SegmentationPipeline
from cv_segment.schemas import PagePayload, TextSpan


def test_pipeline_single_page_stream() -> None:
    pipeline = SegmentationPipeline()
    # Mock parser to return a single page
    mock_page = PagePayload(
        page_index=0,
        width=612.0,
        height=792.0,
        text="Solo Candidate Resume",
        char_count=21,
    )
    with patch.object(pipeline.parser, "parse_pdf", return_value=[mock_page]):
        manifest = pipeline.process_stream("fake_path.pdf", slice_pdf=False)
        assert manifest.total_pages == 1
        assert manifest.total_candidates == 1
        assert len(manifest.boundaries) == 0
        assert manifest.segments[0].page_count == 1


def test_pipeline_multi_candidate_stream() -> None:
    pipeline = SegmentationPipeline()
    # Mock pages: Candidate 1 (p0), Candidate 2 (p1)
    p0 = PagePayload(
        page_index=0,
        width=612.0,
        height=792.0,
        text="Candidate A\nemail@a.com",
        char_count=23,
    )
    p1 = PagePayload(
        page_index=1,
        width=612.0,
        height=792.0,
        text="Candidate B\nemail@b.com",
        char_count=23,
        spans=[
            TextSpan(
                text="email@b.com",
                bbox=(50.0, 50.0, 200.0, 70.0),
                font_name="Arial",
                font_size=12.0,
            )
        ],
    )

    with patch.object(pipeline.parser, "parse_pdf", return_value=[p0, p1]):
        with patch.object(pipeline.heuristics_engine, "evaluate_pair") as mock_heur:
            mock_heur.return_value = MagicMock(short_circuit=True, overall_confidence=0.99)
            manifest = pipeline.process_stream("fake_path.pdf", slice_pdf=False)
            assert manifest.total_pages == 2
            assert manifest.total_candidates == 2
            assert manifest.boundaries == [1]
