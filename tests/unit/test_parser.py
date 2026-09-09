"""Unit tests for PyMuPDF stream parser."""

import pytest

from cv_segment.ingestion.parser import PDFStreamParser


def test_parser_file_not_found() -> None:
    parser = PDFStreamParser()
    with pytest.raises(FileNotFoundError):
        parser.parse_pdf("non_existent_file.pdf")
