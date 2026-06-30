"""Step 3 of the pipeline: map one invoice's text to the standard schema.

This is the only module that talks to the Anthropic API. It hands Claude the
invoice text plus a tool whose input_schema IS the canonical schema (generated
from schemas/invoice.py), and forces a tool call so the reply is always a
structured object in exactly the right shape — no free-text JSON parsing.
"""

from __future__ import annotations

from dataclasses import dataclass

from anthropic import Anthropic

from schemas.invoice import build_tool_schema

TOOL_NAME = "extract_invoice"

_TOOL_DESCRIPTION = (
    "Record every field extracted from a single invoice into the standard "
    "invoice structure. Call this exactly once."
)

_SYSTEM_PROMPT = """You are an expert at reading Indian GST invoices, credit notes and debit notes, and mapping them to a fixed JSON structure regardless of the vendor's layout.

Rules:
- Extract EVERY field that appears anywhere on the invoice into its matching field. If a value is printed, you must find it and map it — do not leave a present value out.
- Use null for any field that is genuinely not present. Use a real null value, never the string "null". Never invent, guess, or carry over values from a different invoice. Empty line-item lists / parties stay empty rather than fabricated.
- Copy identifiers, dates, IRNs, e-Way bill numbers and amounts-in-words EXACTLY as printed (verbatim). Do not reformat or normalise dates.
- For numeric fields output plain numbers with no commas, currency symbols or percent signs (e.g. 113262.75, not "1,13,262.75").
- Indian invoices group digits in the lakh/crore system: "1,43,008.48" means 143008.48 and "18,75,000.00" means 1875000.00. Remove the commas and transcribe the EXACT digits and decimal point — never drop, add, or shift the decimal.
- Never calculate, infer, derive or compute any value. Only transcribe numbers that are actually printed. In particular, per-line cgst_amount / sgst_amount / igst_amount must be null unless the invoice prints a tax amount on that specific line.
- When the goods table has both a tax-inclusive rate column (e.g. "Rate (Incl. of Tax)") and a plain rate column, put the tax-inclusive figure in rate_incl_tax and the tax-exclusive figure in rate. Sanity check: the value you put in rate, multiplied by qty, should approximately equal that line's taxable value/amount.
- round_off is negative when the invoice prints it as "Round Off(-)".
- reverse_charge is true only if the invoice states tax is payable under reverse charge; otherwise false.
- document_type is the heading exactly as printed: TAX INVOICE, CREDIT NOTE, DEBIT NOTE, etc.
- Map parties by role: seller = supplier / "Bill From"; billed_to = buyer / "Bill To"; shipped_to = consignee / "Ship To"; dispatch_from = "Ship From" ONLY when it is distinct from the seller (otherwise leave it null).
- For intra-state invoices fill CGST + SGST; for inter-state invoices fill IGST. Fill per-line tax amounts only when the invoice prints tax per line.
- Map every row of the goods/services table as a separate line item, preserving order."""

# Generous headroom so invoices with many line items / long serial-number lists
# don't truncate. max_tokens caps output only; it does not affect cost, which is
# billed on actual tokens. Stays under the non-streaming HTTP-timeout threshold.
_MAX_TOKENS = 16000


class MappingError(RuntimeError):
    """Raised when the model does not return a structured invoice."""


@dataclass
class MapResult:
    """The extracted invoice data plus the token usage of the call that produced it."""

    data: dict
    input_tokens: int
    output_tokens: int


def build_client(api_key: str) -> Anthropic:
    """Create an Anthropic client. The SDK retries transient errors internally."""
    return Anthropic(api_key=api_key)


def map_invoice(client: Anthropic, model: str, invoice_text: str) -> MapResult:
    """Send one invoice's text to Claude and return the extracted data + usage.

    The data is the tool's input — already structurally aligned with the schema,
    but still passed through the validator afterwards for coercion and sanity
    checks. Token usage is returned alongside it for cost reporting.
    """
    if not invoice_text.strip():
        raise MappingError("Invoice text is empty; nothing to map.")

    tool = {
        "name": TOOL_NAME,
        "description": _TOOL_DESCRIPTION,
        "input_schema": build_tool_schema(),
    }

    response = client.messages.create(
        model=model,
        max_tokens=_MAX_TOKENS,
        system=_SYSTEM_PROMPT,
        tools=[tool],
        tool_choice={"type": "tool", "name": TOOL_NAME},
        messages=[{"role": "user", "content": invoice_text}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == TOOL_NAME:
            return MapResult(
                data=dict(block.input),
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

    raise MappingError("Model did not return an extract_invoice tool call.")
