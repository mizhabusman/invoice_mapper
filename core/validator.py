"""Step 4 of the pipeline: coerce and sanity-check one extracted invoice.

Takes the raw dict from the mapper and:
1. Runs it through the pydantic Invoice model, which cleans numbers (commas,
   currency symbols), fills every missing field with null and drops unexpected
   keys.
2. Performs best-effort arithmetic checks on the totals and records any
   mismatch as a human-readable warning.

Validation NEVER raises and never blocks output — warnings are advisory so a
slightly inconsistent invoice still produces a usable result that a human can
review.
"""

from __future__ import annotations

from pydantic import ValidationError

from schemas.invoice import Invoice

# Rupee tolerance for arithmetic checks (absorbs normal per-line rounding).
_TOLERANCE = 1.0


def _present(*values: object) -> list[float]:
    """Return the given values that are real numbers (not None)."""
    return [float(v) for v in values if isinstance(v, (int, float))]


def _check_totals(invoice: Invoice) -> list[str]:
    """Return warnings for any totals that don't reconcile."""
    warnings: list[str] = []
    t = invoice.totals

    # 1) CGST + SGST + IGST should equal the printed total_tax.
    tax_parts = _present(t.cgst_amount, t.sgst_amount, t.igst_amount)
    if tax_parts and isinstance(t.total_tax, (int, float)):
        tax_sum = sum(tax_parts)
        if abs(tax_sum - float(t.total_tax)) > _TOLERANCE:
            warnings.append(
                f"Tax components ({tax_sum:.2f}) do not match total_tax ({t.total_tax})."
            )

    # 2) taxable + tax + round_off + tcs should equal net_total.
    if isinstance(t.total_taxable, (int, float)) and isinstance(t.net_total, (int, float)):
        tax_total = (
            float(t.total_tax)
            if isinstance(t.total_tax, (int, float))
            else sum(_present(t.cgst_amount, t.sgst_amount, t.igst_amount))
        )
        round_off = float(t.round_off) if isinstance(t.round_off, (int, float)) else 0.0
        tcs = float(t.tcs_amount) if isinstance(t.tcs_amount, (int, float)) else 0.0
        expected_net = float(t.total_taxable) + tax_total + round_off + tcs
        if abs(expected_net - float(t.net_total)) > _TOLERANCE:
            warnings.append(
                f"taxable + tax + round_off ({expected_net:.2f}) does not match "
                f"net_total ({t.net_total})."
            )

    return warnings


def validate_invoice(raw: dict) -> tuple[dict, list[str]]:
    """Coerce a raw extracted dict to the standard shape and check it.

    Returns (clean_invoice_dict, warnings). On the (unexpected) event that
    coercion fails, returns whatever was salvageable plus a warning rather than
    raising, so the pipeline can still emit a result.
    """
    data = dict(raw) if isinstance(raw, dict) else {}
    dropped: list[str] = []
    invoice = None

    # Validate, and if a top-level field is malformed beyond coercion, drop just
    # that field (default fills it) and retry — so one bad field never discards
    # the whole invoice. A handful of passes peels off successive offenders.
    for _ in range(len(data) + 1):
        try:
            invoice = Invoice.model_validate(data)
            break
        except ValidationError as exc:
            offenders = {
                err["loc"][0]
                for err in exc.errors()
                if err["loc"] and isinstance(err["loc"][0], str) and err["loc"][0] in data
            }
            if not offenders:
                break
            for key in offenders:
                data.pop(key, None)
                dropped.append(key)

    if invoice is None:
        return Invoice().model_dump(), ["Could not parse the extracted data."]

    warnings = _check_totals(invoice)
    if dropped:
        warnings.append(
            "Some fields could not be parsed and were left blank: "
            + ", ".join(sorted(set(dropped)))
            + "."
        )
    return invoice.model_dump(), warnings
