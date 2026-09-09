"""API Routes for CV Stream Segmentation Engine."""

import shutil
import tempfile
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from cv_segment.pipeline import SegmentationPipeline

router = APIRouter(prefix="/v1", tags=["Segmentation"])
pipeline = SegmentationPipeline()


@router.get("/health")
def health_check() -> dict[str, str]:
    """Liveness probe for container orchestrator."""
    return {"status": "healthy", "service": "cv-stream-segmenter", "version": "1.0.0"}


@router.post("/segment-stream")
async def segment_pdf_stream(
    file: UploadFile = File(...),
) -> JSONResponse:
    """Accepts a monolithic multi-candidate PDF and returns detected candidate boundary intervals."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        manifest = pipeline.process_stream(tmp_path, slice_pdf=False)
        return JSONResponse(
            content={
                "stream_filename": file.filename,
                "total_pages": manifest.total_pages,
                "total_candidates": manifest.total_candidates,
                "boundaries": manifest.boundaries,
                "segments": [
                    {
                        "candidate_id": s.candidate_id,
                        "start_page": s.start_page,
                        "end_page": s.end_page,
                        "page_count": s.page_count,
                    }
                    for s in manifest.segments
                ],
                "processing_time_ms": manifest.processing_time_ms,
            }
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
