"""Step 2 of the pipeline: group pages into distinct invoices.

A single PDF may contain one invoice, many invoices (e.g. 44 dealer invoices in
one file), or one invoice spread over several pages plus annexures (terms &
conditions, e-Way Bill). This module decides where each invoice begins using
the most reliable signal available, in this order of precedence:

1. IRN (e-invoice reference number) — a globally unique 64-char value printed
   once per invoice and repeated verbatim on that invoice's annexure pages.
   When any page carries an IRN we group purely by it: a new IRN starts a new
   invoice; pages without an IRN attach to the current one.

2. "Page X of N" markers — when there is no IRN but the vendor paginates each
   document (e.g. Redington prints "Page: 1 of 8"). A page numbered 1 starts a
   new invoice; the rest of that run attaches to it.

3. Document header — when neither of the above exists (e.g. a Tally print where
   each page is a self-contained one-page invoice). Each page bearing a
   "TAX INVOICE" / "CREDIT NOTE" header starts a new invoice.

Grouping deliberately does NOT parse the printed invoice number: in
column-flattened text that value is unreliable (it collides with addresses).
The actual invoice_no is recovered later by the mapper from each segment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_IRN_PATTERN = re.compile(
    r"IRN(?:\s*N[O0]|\s*NUMBER)?\s*[:#]?\s*([0-9A-Fa-f][0-9A-Fa-f \-\r\n]{30,100})",
    re.IGNORECASE,
)
_PAGE_MARKER_PATTERN = re.compile(r"\bPAGE\s*:?\s*(\d+)\s+of\s+(\d+)\b", re.IGNORECASE)
_HEADER_PATTERN = re.compile(
    r"\b(TAX\s+INVOICE|CREDIT\s+NOTE|DEBIT\s+NOTE|BILL\s+OF\s+SUPPLY)\b", re.IGNORECASE
)
_HEX = re.compile(r"[^0-9A-Fa-f]")


@dataclass
class InvoiceSegment:
    """One invoice's worth of pages and combined text."""

    pages: list[int] = field(default_factory=list)  # 1-based page numbers
    text: str = ""  # all pages joined (full, untrimmed — kept for provenance)
    page_texts: list[str] = field(default_factory=list)  # per-page text, same order


# --- Boilerplate detection (used to trim dead pages before sending to the AI) --
_TC_MARKER = re.compile(
    r"TERMS\s+AND\s+CONDITIONS|TERMS\s*&\s*CONDITIONS|TERMS\s+OF\s+SALE", re.IGNORECASE
)
_AMOUNT = re.compile(r"\d[\d,]*\.\d{2}(?!\d)")  # a currency-style amount, e.g. 1,234.56
_STRONG_DATA = re.compile(r"GSTIN|\bHSN\b|\bSAC\b|TAXABLE\s+VALUE|\bIRN\b", re.IGNORECASE)


def is_boilerplate_page(text: str) -> bool:
    """True only for a page that is clearly pure terms & conditions / legal prose.

    Deliberately conservative: a page is dropped ONLY if it has a T&C heading AND
    contains no currency amounts AND no GSTIN/HSN/IRN. Any real data page has
    amounts, so this can never drop a page that carries invoice data.
    """
    if not _TC_MARKER.search(text):
        return False
    if _AMOUNT.search(text) or _STRONG_DATA.search(text):
        return False
    return True


def text_for_mapping(segment: "InvoiceSegment") -> str:
    """Return the segment text to send to the model, with dead T&C pages removed.

    Falls back to the full text whenever trimming isn't safe (single-page
    segment, nothing to drop, or everything would be dropped) — so the worst
    case is identical to sending the untrimmed text.
    """
    if len(segment.page_texts) <= 1:
        return segment.text
    kept = [t for t in segment.page_texts if not is_boilerplate_page(t)]
    if not kept or len(kept) == len(segment.page_texts):
        return segment.text
    return "\n\n".join(kept)


def find_irn(page_text: str) -> str | None:
    """Return the normalised 64-char IRN on a page, or None.

    Handles IRNs that wrap across a line with a hyphen by stripping every
    non-hex character and keeping the leading 64 hex chars. Used only as a
    grouping key, so consistency matters more than absolute exactness.
    """
    match = _IRN_PATTERN.search(page_text)
    if not match:
        return None
    hex_only = _HEX.sub("", match.group(1))[:64]
    return hex_only if len(hex_only) >= 32 else None


def starts_new_document(page_text: str) -> bool:
    """True if the page is 'Page 1 of N' — i.e. the first page of a document."""
    match = _PAGE_MARKER_PATTERN.search(page_text)
    return bool(match) and match.group(1) == "1"


def has_page_marker(page_text: str) -> bool:
    """True if the page carries any 'Page X of N' marker."""
    return bool(_PAGE_MARKER_PATTERN.search(page_text))


def has_title_header(page_text: str, max_lines: int = 3) -> bool:
    """True if a document-type header appears near the TOP of the page.

    A title at the top (e.g. first line "TAX INVOICE" / "Credit Note") marks the
    start of a new document. The same words appearing lower down are a running
    header / footer on a continuation page and must NOT start a new invoice —
    that is what keeps a 2-page credit note from being split in two.
    """
    top_lines = [line for line in page_text.splitlines() if line.strip()][:max_lines]
    return bool(_HEADER_PATTERN.search("\n".join(top_lines)))


def _append(segments: list[InvoiceSegment], current: InvoiceSegment, page_no: int, text: str) -> None:
    current.pages.append(page_no)
    current.page_texts.append(text)
    current.text = text if not current.text else f"{current.text}\n\n{text}"


def _group(pages: list[str], is_boundary) -> list[InvoiceSegment]:
    """Walk pages, opening a new segment whenever is_boundary(index) is True.

    The first page always opens the first segment.
    """
    segments: list[InvoiceSegment] = []
    current: InvoiceSegment | None = None
    for index, text in enumerate(pages):
        if current is None or is_boundary(index):
            current = InvoiceSegment()
            segments.append(current)
        _append(segments, current, index + 1, text)
    return segments


def split_invoices(pages: list[str]) -> list[InvoiceSegment]:
    """Group page texts into one InvoiceSegment per distinct invoice."""
    if not pages:
        return []

    # 1) Group by IRN when the document is an e-invoice.
    irns = [find_irn(text) for text in pages]
    if any(irns):
        last_seen = irns[0]

        def irn_boundary(index: int) -> bool:
            nonlocal last_seen
            key = irns[index]
            if key is None:
                return False  # annexure / continuation page keeps current IRN
            is_new = key != last_seen
            last_seen = key
            return is_new

        return _group(pages, irn_boundary)

    # 2) Group by "Page 1 of N" markers when the vendor paginates documents.
    if any(has_page_marker(text) for text in pages):
        return _group(pages, lambda index: starts_new_document(pages[index]))

    # 3) Otherwise treat each page whose title header sits at the top as a new
    #    invoice (continuation pages repeat the header lower down and are merged).
    return _group(pages, lambda index: has_title_header(pages[index]))
