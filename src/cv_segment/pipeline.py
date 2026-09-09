"""End-to-End CV Stream Segmentation Pipeline Orchestrator.

Assigned Engineer: Engineer 1 (User)
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Union

from cv_segment.features.pairwise import PairwiseFeatureExtractor
from cv_segment.heuristics.engine import HeuristicGatingEngine
from cv_segment.ingestion.parser import PDFStreamParser
from cv_segment.models.classifier import LightGBMPairwiseClassifier
from cv_segment.models.sequence_opt import ViterbiSequenceOptimizer
from cv_segment.schemas import CandidateSegment, SegmentationManifest
from cv_segment.slicer.pdf_slicer import PikePDFSlicer


class SegmentationPipeline:
    """Master orchestrator connecting Ingestion -> Heuristics -> LightGBM -> Viterbi -> Slicer."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        output_dir: Union[str, Path] = "output_candidates",
    ) -> None:
        self.parser = PDFStreamParser()
        self.heuristics_engine = HeuristicGatingEngine()
        self.feature_extractor = PairwiseFeatureExtractor()
        self.classifier = LightGBMPairwiseClassifier(model_path)
        self.sequence_optimizer = ViterbiSequenceOptimizer()
        self.slicer = PikePDFSlicer(output_dir)

    def process_stream(
        self,
        pdf_path: Union[str, Path],
        slice_pdf: bool = True,
    ) -> SegmentationManifest:
        """Processes a merged PDF stream and outputs segmentation boundaries and candidate files."""
        start_time = time.perf_counter()
        source_path = Path(pdf_path)

        # Tier 0: Ingestion
        pages = self.parser.parse_pdf(source_path)
        total_pages = len(pages)

        if total_pages <= 1:
            # Single-page PDF has no internal boundaries
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            initial_segments = [
                CandidateSegment(
                    candidate_id="candidate_001",
                    start_page=1,
                    end_page=total_pages,
                    page_count=total_pages,
                    output_filename=str(source_path),
                )
            ]
            return SegmentationManifest(
                stream_filename=source_path.name,
                total_pages=total_pages,
                total_candidates=1,
                boundaries=[],
                segments=initial_segments,
                processing_time_ms=elapsed_ms,
            )

        # Pairwise Emission Probabilities
        pairwise_split_probs: list[float] = []
        cumulative_pages = 1

        for i in range(total_pages - 1):
            page_i = pages[i]
            page_j = pages[i + 1]

            # Tier 1: Deterministic Heuristic Gating
            heuristic_res = self.heuristics_engine.evaluate_pair(page_i, page_j)

            if heuristic_res.short_circuit:
                # Fast-path decision
                pairwise_split_probs.append(heuristic_res.overall_confidence)
                if heuristic_res.overall_confidence >= 0.5:
                    cumulative_pages = 1
                else:
                    cumulative_pages += 1
            else:
                # Tier 2: Classical ML (LightGBM on 36 pairwise features)
                features = self.feature_extractor.extract_features(
                    page_i, page_j, cumulative_pages_in_segment=cumulative_pages
                )
                prob = self.classifier.predict_split_probability(features)
                pairwise_split_probs.append(prob)
                if prob >= 0.5:
                    cumulative_pages = 1
                else:
                    cumulative_pages += 1

        # Tier 3: Global Sequence Trellis Optimization (Viterbi HMM)
        boundary_cuts = self.sequence_optimizer.decode(pairwise_split_probs)

        # Lossless Slicing
        segments: list[CandidateSegment] = []
        if slice_pdf:
            segments = self.slicer.slice_stream(source_path, boundary_cuts)
        else:
            # Generate segment metadata without writing PDF slices
            intervals: list[tuple[int, int]] = []
            curr = 1
            for cut in boundary_cuts:
                intervals.append((curr, cut))
                curr = cut + 1
            if curr <= total_pages:
                intervals.append((curr, total_pages))

            for idx, (s_p, e_p) in enumerate(intervals, start=1):
                segments.append(
                    CandidateSegment(
                        candidate_id=f"candidate_{idx:03d}",
                        start_page=s_p,
                        end_page=e_p,
                        page_count=e_p - s_p + 1,
                    )
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        manifest = SegmentationManifest(
            stream_filename=source_path.name,
            total_pages=total_pages,
            total_candidates=len(segments),
            boundaries=boundary_cuts,
            segments=segments,
            processing_time_ms=elapsed_ms,
        )

        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="CV Stream Segmentation CLI")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input merged PDF file.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("output_candidates"),
        help="Directory to save candidate PDF slices.",
    )
    parser.add_argument(
        "--no-slice", action="store_true", help="Only predict boundaries without writing slices."
    )
    args = parser.parse_args()

    pipeline = SegmentationPipeline(output_dir=args.output)
    print(f"[*] Processing stream: {args.input}")
    manifest = pipeline.process_stream(args.input, slice_pdf=not args.no_slice)

    print("\n================ SEGMENTATION SUMMARY ================")
    print(f"File: {manifest.stream_filename}")
    print(f"Total Pages: {manifest.total_pages}")
    print(f"Total Candidates Detected: {manifest.total_candidates}")
    print(f"Boundaries (Cut points): {manifest.boundaries}")
    print(f"Processing Time: {manifest.processing_time_ms:.2f} ms")
    print("======================================================")

    # Save manifest.json
    manifest_path = args.output / "manifest.json"
    args.output.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "stream_filename": manifest.stream_filename,
                "total_pages": manifest.total_pages,
                "total_candidates": manifest.total_candidates,
                "boundaries": manifest.boundaries,
                "processing_time_ms": manifest.processing_time_ms,
            },
            f,
            indent=2,
        )
    print(f"[*] Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
