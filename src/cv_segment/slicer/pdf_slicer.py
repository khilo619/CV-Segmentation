"""Lossless, Zero-Copy PDF Slicing Engine via pikepdf (QPDF bindings).

Assigned Engineer: Engineer 1 (User)
"""

from pathlib import Path
from typing import Union

try:
    import pikepdf
except ImportError:
    pikepdf = None  # type: ignore[assignment]

from cv_segment.schemas import CandidateSegment


class PikePDFSlicer:
    """Slices monolithic multi-document PDF into individual candidate PDF files."""

    def __init__(self, output_dir: Union[str, Path]) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def slice_stream(
        self,
        source_pdf: Union[str, Path],
        boundary_cut_points: list[int],
    ) -> list[CandidateSegment]:
        """Slices PDF stream according to 1-indexed boundary cut points.

        Args:
            source_pdf: Path to the monolithic PDF.
            boundary_cut_points: List of 1-indexed page cut points.
                e.g., [2, 5] in a 7-page document splits into:
                - Candidate 1: pages 1 to 2
                - Candidate 2: pages 3 to 5
                - Candidate 3: pages 6 to 7
        """
        source_path = Path(source_pdf)
        if not source_path.exists():
            raise FileNotFoundError(f"Source PDF not found: {source_pdf}")

        if pikepdf is None:
            raise RuntimeError("pikepdf is not installed. Run `pip install pikepdf`.")

        segments: list[CandidateSegment] = []

        with pikepdf.Pdf.open(str(source_path)) as src:
            total_pages = len(src.pages)
            cut_points = sorted(set(boundary_cut_points))

            # Build list of (start_page_1idx, end_page_1idx) intervals
            intervals: list[tuple[int, int]] = []
            curr_start = 1

            for cut in cut_points:
                if curr_start <= cut <= total_pages:
                    intervals.append((curr_start, cut))
                    curr_start = cut + 1

            if curr_start <= total_pages:
                intervals.append((curr_start, total_pages))

            # Slice intervals into discrete PDF files
            for idx, (start_p, end_p) in enumerate(intervals, start=1):
                candidate_id = f"candidate_{idx:03d}"
                out_filename = f"{candidate_id}.pdf"
                out_path = self.output_dir / out_filename

                new_pdf = pikepdf.Pdf.new()
                # 0-indexed page slice
                for page_num in range(start_p - 1, end_p):
                    new_pdf.pages.append(src.pages[page_num])

                new_pdf.save(str(out_path))

                segments.append(
                    CandidateSegment(
                        candidate_id=candidate_id,
                        start_page=start_p,
                        end_page=end_p,
                        page_count=end_p - start_p + 1,
                        output_filename=str(out_path),
                    )
                )

        return segments
