"""Data schemas and internal communication contracts for the CV segmentation engine."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextSpan:
    """Represents a single text span with typography and bounding box."""

    text: str
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    font_name: str
    font_size: float
    flags: int = 0


@dataclass
class PagePayload:
    """Standardized representation of a single extracted PDF page."""

    page_index: int  # 0-indexed page index in the stream
    width: float
    height: float
    text: str
    char_count: int
    spans: list[TextSpan] = field(default_factory=list)
    has_images: bool = False
    is_scanned: bool = False


@dataclass
class HeuristicEvaluation:
    """Output of the Tier 1 deterministic rules engine for an adjacent page pair."""

    pair_index: int  # Transition between page_i and page_{i+1}
    pii_score: float
    font_dominance_ratio: float
    is_explicit_boundary: bool
    is_explicit_continuation: bool
    syntactic_continuation_score: float
    overall_confidence: float  # Estimated boundary probability [0.0, 1.0]
    short_circuit: bool = False  # True if confidence >= 0.98 or <= 0.02


@dataclass
class BoundaryDecision:
    """Final decision for an adjacent page transition."""

    pair_index: int  # Boundary between page_index and page_index + 1
    is_boundary: bool
    split_probability: float
    decision_source: str  # "heuristic_fastpath" | "lightgbm" | "viterbi_hmm"


@dataclass
class CandidateSegment:
    """Represents an isolated, segmented candidate CV."""

    candidate_id: str
    start_page: int  # 1-indexed for human / recruiter readability
    end_page: int  # 1-indexed, inclusive
    page_count: int
    detected_name: Optional[str] = None
    output_filename: Optional[str] = None


@dataclass
class SegmentationManifest:
    """Top-level manifest emitted after segmenting a stream."""

    stream_filename: str
    total_pages: int
    total_candidates: int
    boundaries: list[int]  # 1-indexed split cut points
    segments: list[CandidateSegment]
    processing_time_ms: float
