"""Unit tests for pikepdf lossless slicer."""

import pytest

from cv_segment.slicer.pdf_slicer import PikePDFSlicer


def test_slicer_file_not_found(tmp_path: pytest.TempPathFactory) -> None:
    slicer = PikePDFSlicer(output_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        slicer.slice_stream("non_existent_stream.pdf", [1])
