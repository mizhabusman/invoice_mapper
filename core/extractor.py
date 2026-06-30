"""Step 1 of the pipeline: PDF bytes -> text, one string per page.

Uses pdfplumber, which preserves layout reasonably well for the tabular
invoices this tool targets. Pages with no extractable text (e.g. a scanned
image) come back as empty strings; the pipeline decides what to do with that.
"""

from __future__ import annotations

import io

import pdfplumber


def extract_pages(pdf_bytes: bytes) -> list[str]:
    """Return the text of each page in order.

    The returned list is positional: index 0 is page 1, and so on. Layout is
    preserved via pdfplumber's default text extraction so that columnar tables
    stay readable for the language model.
    """
    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=1.5, y_tolerance=3) or ""
            pages.append(text)
    return pages


def has_text(pages: list[str]) -> bool:
    """True if at least one page yielded non-whitespace text."""
    return any(page.strip() for page in pages)
