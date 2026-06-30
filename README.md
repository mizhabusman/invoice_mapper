# Invoice Mapper

Upload a text-extractable invoice PDF (any vendor layout) and get clean,
standardised JSON. Handles single-invoice and multi-invoice PDFs, intra-state
(CGST+SGST) and inter-state (IGST) tax, credit/debit notes, and annexures like
e-Way Bills. Mapping is done by Claude via a fixed structured schema.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt

cp .env.example .env            # then edit .env and add your ANTHROPIC_API_KEY
```

## Run the demo UI

```bash
streamlit run app.py
```

Upload a PDF, preview the result, and download it — a single `.json` for a
one-invoice PDF, or a `.zip` of per-invoice files for a multi-invoice PDF.

## Use the pipeline directly (any frontend)

```python
from pipeline import process_pdf, to_download

result = process_pdf(pdf_bytes, "invoice.pdf")     # ProcessResult
records = result.invoices                           # list[dict], schema-shaped
print(result.usage.cost_usd, result.usage.input_tokens, result.usage.output_tokens)
data, filename, mime = to_download(records)         # bytes ready to save/serve
```

## Architecture

```
app.py            Streamlit UI (presentation only)
pipeline.py       Orchestration + output helpers (the entry point)
core/
  extractor.py    PDF -> per-page text (pdfplumber)
  splitter.py     Group pages into distinct invoices
  mapper.py       Claude call -> structured JSON (only API caller)
  validator.py    Coercion + totals sanity checks
schemas/
  invoice.py      Canonical schema (drives both the tool and validation)
config.py         Loads .env settings
```

All logic is UI-agnostic and lives in `core/` + `pipeline.py`. Swapping
Streamlit for a REST API or web frontend means rewriting only `app.py`.

## Output shape

Every record follows `schemas/invoice.py`, plus two added keys:
- `_source` — origin file, 1-based invoice index, and source page numbers.
- `_warnings` — advisory notes when totals don't reconcile (never blocks output).

Fields that don't exist on a given invoice are `null`, so every record has the
same shape regardless of vendor.

## Notes / limitations

- Text-extractable PDFs only for now; scanned/image PDFs (OCR) are a future
  addition and are reported with a clear message.
- Dates and identifiers are copied verbatim from the invoice (not reformatted).
```
