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


def _num(value: object) -> float | None:
    """Return value as float if it is a real number, else None."""
    return float(value) if isinstance(value, (int, float)) else None


def _check_lines(invoice: Invoice) -> list[str]:
    """Best-effort per-line arithmetic checks (warnings only, data unchanged).

    Catches gross mismatches such as a swapped rate/rate-incl column or a
    misplaced decimal — errors that reconcile at the invoice level and would
    otherwise pass silently.
    """
    warnings: list[str] = []
    lines = invoice.line_items

    # rate x qty should reconcile with the line's taxable value or amount.
    bad: list = []
    for position, li in enumerate(lines, start=1):
        qty = _num(li.qty)
        discount = _num(li.discount)
        if not qty or (discount and discount != 0):
            continue  # skip zero/None qty and discounted lines (taxable != rate*qty)
        rates = [r for r in (_num(li.rate), _num(li.rate_incl_tax)) if r is not None]
        targets = [t for t in (_num(li.taxable_value), _num(li.amount)) if t is not None]
        if not rates or not targets:
            continue
        # OK if ANY rate x qty matches ANY target within 2% (absorbs rounding,
        # tax-inclusive vs exclusive, small discounts).
        reconciles = any(
            abs(r * qty - t) <= max(1.0, 0.02 * abs(t)) for r in rates for t in targets
        )
        if not reconciles:
            bad.append(_num(li.sno) or position)

    if bad:
        labels = ", ".join(str(int(b)) for b in bad)
        warnings.append(
            f"Line(s) {labels}: rate x qty does not match the line amount — verify rate/qty."
        )

    # Sum of per-line taxable values should match the invoice total_taxable.
    line_taxables = [_num(li.taxable_value) for li in lines]
    total_taxable = _num(invoice.totals.total_taxable)
    if lines and total_taxable is not None and all(t is not None for t in line_taxables):
        line_sum = sum(line_taxables)  # type: ignore[arg-type]
        if abs(line_sum - total_taxable) > max(1.0, 0.01 * abs(total_taxable)):
            warnings.append(
                f"Sum of line taxable values ({line_sum:.2f}) does not match "
                f"total_taxable ({invoice.totals.total_taxable})."
            )

    return warnings


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

    warnings = _check_totals(invoice) + _check_lines(invoice)
    if dropped:
        warnings.append(
            "Some fields could not be parsed and were left blank: "
            + ", ".join(sorted(set(dropped)))
            + "."
        )
    return invoice.model_dump(), warnings
