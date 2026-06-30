"""UI-agnostic orchestration: PDF bytes in, standard invoice records out.

This is the single entry point every frontend should call. Swapping Streamlit
for a REST API or a different UI means rewriting only the presentation layer —
the contract here (`process_pdf`) and the output helpers stay the same.

Flow: extract pages -> split into invoices -> map each with Claude (in
parallel) -> validate -> attach provenance (`_source`) and diagnostics
(`_warnings`).
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional

from config import Settings, estimate_cost, load_settings
from core.extractor import extract_pages, has_text
from core.mapper import build_client, map_invoice
from core.splitter import InvoiceSegment, split_invoices
from core.validator import validate_invoice

# Optional callback invoked as (completed_count, total_count) for UI progress.
ProgressCallback = Callable[[int, int], None]


class PipelineError(RuntimeError):
    """Raised for input problems the caller should surface to the user."""


@dataclass
class Usage:
    """Aggregated token spend and estimated cost for a processing run."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = None  # None when the model has no known pricing
    cost_inr: Optional[float] = None  # cost_usd converted at the configured rate


@dataclass
class ProcessResult:
    """The outcome of processing one PDF: the invoice records plus run usage."""

    invoices: list[dict] = field(default_factory=list)
    usage: Optional[Usage] = None


def _process_segment(
    client, model: str, segment: InvoiceSegment, index: int, filename: str
) -> tuple[dict, int, int]:
    """Map + validate one segment; return the record and its token usage."""
    mapped = map_invoice(client, model, segment.text)
    clean, warnings = validate_invoice(mapped.data)
    clean["_source"] = {
        "file": filename,
        "invoice_index": index,  # 1-based position within the file
        "pages": segment.pages,
    }
    clean["_warnings"] = warnings
    return clean, mapped.input_tokens, mapped.output_tokens


def process_pdf(
    pdf_bytes: bytes,
    filename: str,
    settings: Optional[Settings] = None,
    progress: Optional[ProgressCallback] = None,
) -> ProcessResult:
    """Convert a text-extractable PDF into standard invoice records + usage.

    Raises PipelineError when the API key is missing or the PDF has no
    extractable text (e.g. a scanned image). Each record matches the schema
    plus `_source` and `_warnings`; `usage` carries the run's token spend and
    estimated cost.
    """
    settings = settings or load_settings()
    if not settings.is_api_key_present:
        raise PipelineError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    try:
        pages = extract_pages(pdf_bytes)
    except Exception as exc:  # corrupt / password-protected / not a real PDF
        raise PipelineError(
            "Could not read this PDF — it may be corrupted, password-protected, "
            "or not a valid PDF."
        ) from exc
    if not has_text(pages):
        raise PipelineError(
            "No extractable text found. This PDF looks scanned/image-based, which "
            "is not supported yet (text-extractable PDFs only for now)."
        )

    segments = split_invoices(pages)
    if not segments:
        raise PipelineError("Could not identify any invoice in this PDF.")

    client = build_client(settings.anthropic_api_key)
    total = len(segments)
    results: list[Optional[tuple[dict, int, int]]] = [None] * total
    completed = 0

    workers = max(1, min(settings.max_concurrency, total))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_segment, client, settings.model, segment, i + 1, filename
            ): i
            for i, segment in enumerate(segments)
        }
        # Collect as they finish to drive progress, but place by original index.
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            completed += 1
            if progress:
                progress(completed, total)

    # No result can be None here (any exception would have propagated).
    invoices = [r[0] for r in results if r is not None]
    input_tokens = sum(r[1] for r in results if r is not None)
    output_tokens = sum(r[2] for r in results if r is not None)
    cost_usd = estimate_cost(settings.model, input_tokens, output_tokens)
    usage = Usage(
        model=settings.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        cost_inr=cost_usd * settings.usd_to_inr if cost_usd is not None else None,
    )
    return ProcessResult(invoices=invoices, usage=usage)


# --- Output helpers (reusable by any frontend) --------------------------------

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _to_json(record: dict) -> str:
    """Serialise one record to pretty, UTF-8-friendly JSON."""
    return json.dumps(record, ensure_ascii=False, indent=2)


def result_filename(record: dict, index: int) -> str:
    """Build a safe .json filename for a record, based on its invoice number.

    Falls back to the 1-based index when no invoice number was extracted.
    """
    invoice_no = record.get("invoice_no")
    base = _SAFE_NAME.sub("_", str(invoice_no).strip()) if invoice_no else ""
    if not base:
        base = f"invoice_{index}"
    return f"{base}.json"


def to_download(records: list[dict]) -> tuple[bytes, str, str]:
    """Package results for download.

    Returns (data_bytes, filename, mime_type):
    - a single invoice -> one .json file
    - multiple invoices -> a .zip of individual .json files with unique names
    """
    if len(records) == 1:
        data = _to_json(records[0]).encode("utf-8")
        return data, result_filename(records[0], 1), "application/json"

    buffer = io.BytesIO()
    used: set[str] = set()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for i, record in enumerate(records, start=1):
            name = result_filename(record, i)
            if name in used:  # guarantee uniqueness on duplicate invoice numbers
                name = f"{name[:-5]}_{i}.json"
            used.add(name)
            archive.writestr(name, _to_json(record))
    return buffer.getvalue(), "invoices.zip", "application/zip"
