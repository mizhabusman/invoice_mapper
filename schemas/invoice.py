"""Standard invoice schema — the single source of truth.

These pydantic models define the canonical JSON shape the tool produces. The
same models are used for two purposes:

1. Generating the Anthropic tool input_schema (`build_tool_schema`), so Claude
   is forced to return data in exactly this shape.
2. Coercing/validating Claude's raw output (see core/validator.py).

Design notes:
- Every field is Optional and defaults to None (lists default to empty). A field
  that does not exist on a given invoice simply comes back null, so a simple
  5-column invoice and a 17-column one produce the same shape.
- Dates and identifiers are kept as strings exactly as printed on the invoice.
  We deliberately do NOT reformat dates — that would risk corrupting data.
- Numbers are cleaned (commas / currency symbols / percent signs stripped) and
  returned as int when integral, float otherwise, via the shared `Number` type.
- `_source` (provenance) and `_warnings` (validation notes) are NOT part of this
  model. They are added by the pipeline so this model stays a pure description
  of what Claude extracts.
"""

from __future__ import annotations

from typing import Annotated, Optional, Union

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

_NULL_TOKENS = {"", "-", "na", "n/a", "null", "none", "nil"}


def clean_number(value: object) -> Optional[Union[int, float]]:
    """Normalise a numeric value coming from Claude or a raw string.

    Handles Indian-formatted numbers ("1,13,262.75"), currency symbols, percent
    signs and stray whitespace. Returns an int when the result is integral
    (so quantities/percentages render as 9 not 9.0), a float otherwise, or None
    for blanks and explicit null tokens. Never raises — unparseable input
    becomes None so a single bad cell can't fail the whole invoice.
    """
    if value is None:
        return None
    if isinstance(value, bool):  # guard: bools are ints in Python
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        if text.lower() in _NULL_TOKENS:
            return None
        # Strip everything except digits, sign and decimal point.
        cleaned = (
            text.replace(",", "")
            .replace("₹", "")  # ₹
            .replace("Rs", "")
            .replace("rs", "")
            .replace("%", "")
            .replace("INR", "")
            .strip()
        )
        try:
            number = float(cleaned)
        except ValueError:
            return None
    return int(number) if number.is_integer() else number


# Shared numeric type: cleans input, allows int or float, nullable.
Number = Annotated[Optional[Union[int, float]], BeforeValidator(clean_number)]


def _coerce_model(value: object) -> object:
    """Normalise a nested-object field before validation.

    Models (haiku especially) sometimes emit the literal string "null" or an
    actual null for an absent sub-object. Map those to an empty object so the
    field becomes an all-null model instead of failing the whole record.
    """
    if value is None:
        return {}
    if isinstance(value, str):
        return {} if value.strip().lower() in _NULL_TOKENS else value
    return value


def _coerce_list(value: object) -> object:
    """Map a null-ish value for a list field to an empty list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [] if value.strip().lower() in _NULL_TOKENS else value
    return value


class _Model(BaseModel):
    """Base that ignores unexpected keys instead of failing."""

    model_config = ConfigDict(extra="ignore")


class Party(_Model):
    """A party on the invoice: seller, buyer, consignee or dispatch origin.

    A single superset model covers every party type; fields that don't apply to
    a given party simply stay null (e.g. a seller has no place_of_supply).
    """

    name: Optional[str] = Field(None, description="Legal/registered name of the party.")
    address: Optional[str] = Field(None, description="Street address, single line.")
    city_pincode: Optional[str] = Field(
        None, description="City and PIN code, e.g. 'VIJAYAWADA.-520012'."
    )
    gstin: Optional[str] = Field(None, description="GSTIN / UIN of the party.")
    pan: Optional[str] = Field(None, description="PAN, if printed.")
    state_code: Optional[str] = Field(None, description="GST state code, e.g. '37'.")
    state_name: Optional[str] = Field(None, description="State name, e.g. 'ANDHRAPRADESH'.")
    place_of_supply: Optional[str] = Field(
        None, description="Place of supply, if printed against this party."
    )
    phone: Optional[str] = Field(None, description="Phone/contact number, if printed.")
    email: Optional[str] = Field(None, description="Email address, if printed.")
    cin: Optional[str] = Field(None, description="Corporate Identity Number, if printed.")


class LineItem(_Model):
    """One row in the goods/services table."""

    sno: Number = Field(None, description="Serial number of the line as printed.")
    item_code: Optional[str] = Field(
        None, description="Item/SKU/part code, if present (e.g. Lava item code, OnePlus SKU)."
    )
    particulars: Optional[str] = Field(
        None, description="Full description of the goods/service, verbatim."
    )
    hsn_sac: Optional[str] = Field(None, description="HSN or SAC code.")
    qty: Number = Field(None, description="Quantity.")
    unit: Optional[str] = Field(None, description="Unit of measure, e.g. EA, NO, PCS.")
    rate: Number = Field(None, description="Rate per unit, exclusive of tax.")
    rate_incl_tax: Number = Field(
        None, description="Rate per unit INCLUDING tax, only if the invoice prints it."
    )
    discount: Number = Field(None, description="Discount amount on this line, if any.")
    taxable_value: Number = Field(
        None, description="Taxable value for this line (after discount), if printed separately."
    )
    gst_pct: Number = Field(None, description="Total GST rate % applied to the line.")
    cgst_amount: Number = Field(None, description="CGST amount for this line, if printed per line.")
    sgst_amount: Number = Field(None, description="SGST amount for this line, if printed per line.")
    igst_amount: Number = Field(None, description="IGST amount for this line, if printed per line.")
    amount: Number = Field(
        None, description="Line amount/extended amount as printed in the rightmost amount column."
    )
    serial_numbers: Annotated[list[str], BeforeValidator(_coerce_list)] = Field(
        default_factory=list, description="Product serial numbers listed for this line, if any."
    )


class Totals(_Model):
    """Invoice-level totals and tax summary."""

    total_taxable: Number = Field(None, description="Sum of taxable value across all lines.")
    cgst_pct: Number = Field(None, description="CGST rate %, if intra-state.")
    cgst_amount: Number = Field(None, description="Total CGST amount.")
    sgst_pct: Number = Field(None, description="SGST rate %, if intra-state.")
    sgst_amount: Number = Field(None, description="Total SGST amount.")
    igst_pct: Number = Field(None, description="IGST rate %, if inter-state.")
    igst_amount: Number = Field(None, description="Total IGST amount.")
    total_tax: Number = Field(None, description="Total tax amount (CGST+SGST or IGST), if printed.")
    tcs_amount: Number = Field(None, description="TCS amount, if charged.")
    sub_total: Number = Field(None, description="Sub total (taxable + tax) before round off.")
    round_off: Number = Field(
        None, description="Round-off adjustment. Negative when shown as 'Round Off(-)'."
    )
    net_total: Number = Field(None, description="Final payable/net total of the invoice.")
    total_qty: Number = Field(None, description="Total quantity across all lines.")


class BankDetails(_Model):
    bank: Optional[str] = Field(None, description="Bank name.")
    branch: Optional[str] = Field(None, description="Branch name.")
    account_no: Optional[str] = Field(None, description="Account number.")
    ifsc: Optional[str] = Field(None, description="IFSC code.")
    account_type: Optional[str] = Field(None, description="Account type, e.g. 'CA A/C'.")
    swift: Optional[str] = Field(None, description="SWIFT code, if printed.")


class EInvoice(_Model):
    irn: Optional[str] = Field(None, description="Invoice Reference Number (IRN).")
    ack_no: Optional[str] = Field(None, description="Acknowledgement number.")
    ack_date: Optional[str] = Field(None, description="Acknowledgement date, verbatim.")
    category: Optional[str] = Field(None, description="Category, e.g. 'B2B'.")


class EwayBill(_Model):
    number: Optional[str] = Field(None, description="e-Way Bill number.")
    ack_date: Optional[str] = Field(None, description="e-Way Bill acknowledgement date, verbatim.")
    valid_from: Optional[str] = Field(None, description="Validity start / generated date, verbatim.")
    valid_till: Optional[str] = Field(None, description="Validity end date, verbatim.")
    distance_km: Number = Field(None, description="Approx distance in KM, if printed.")
    vehicle_no: Optional[str] = Field(None, description="Vehicle number on the e-Way bill.")
    mode: Optional[str] = Field(None, description="Transport mode, e.g. '1 - Road'.")


class Transport(_Model):
    transporter_name: Optional[str] = Field(None, description="Transporter name/agency.")
    vehicle_no: Optional[str] = Field(None, description="Vehicle number.")
    mode: Optional[str] = Field(None, description="Transportation mode, e.g. 'Road'.")
    lr_no: Optional[str] = Field(None, description="LR number, if printed.")
    carrier: Optional[str] = Field(None, description="Carrier / consignment note details.")


class Invoice(_Model):
    """The full standard invoice record extracted from one invoice."""

    document_type: Optional[str] = Field(
        None, description="Document type exactly as printed: TAX INVOICE, CREDIT NOTE, DEBIT NOTE, etc."
    )
    copy_type: Optional[str] = Field(
        None, description="Copy designation, e.g. 'Original For Recipient'."
    )
    invoice_no: Optional[str] = Field(None, description="Invoice / document number.")
    invoice_date: Optional[str] = Field(None, description="Invoice date, copied verbatim.")
    due_date: Optional[str] = Field(None, description="Payment due date, verbatim, if printed.")
    reference_no: Optional[str] = Field(
        None, description="Any reference number (Ref No, original invoice ref, etc.)."
    )

    seller: Annotated[Party, BeforeValidator(_coerce_model)] = Field(
        default_factory=Party, description="The supplier / seller (Bill From)."
    )
    billed_to: Annotated[Party, BeforeValidator(_coerce_model)] = Field(
        default_factory=Party, description="The buyer (Bill To)."
    )
    shipped_to: Annotated[Party, BeforeValidator(_coerce_model)] = Field(
        default_factory=Party, description="The consignee (Ship To)."
    )
    dispatch_from: Annotated[Party, BeforeValidator(_coerce_model)] = Field(
        default_factory=Party,
        description="Dispatch origin (Ship From), only when distinct from the seller.",
    )

    line_items: Annotated[list[LineItem], BeforeValidator(_coerce_list)] = Field(
        default_factory=list, description="Every row in the goods/services table."
    )
    totals: Annotated[Totals, BeforeValidator(_coerce_model)] = Field(
        default_factory=Totals, description="Invoice totals and tax summary."
    )

    amount_in_words: Optional[str] = Field(None, description="Total amount written in words.")
    notes: Optional[str] = Field(None, description="Free-text notes printed on the invoice.")
    po_reference: Optional[str] = Field(None, description="Purchase order number/reference.")
    po_date: Optional[str] = Field(None, description="Purchase order date, verbatim.")
    payment_terms: Optional[str] = Field(
        None, description="Payment terms, e.g. '35 Days', '30 DAYS FROM'."
    )
    reverse_charge: Optional[bool] = Field(
        None, description="True if tax is payable under reverse charge, else False."
    )
    place_of_supply: Optional[str] = Field(
        None, description="Overall place of supply for the invoice."
    )

    bank_details: Annotated[BankDetails, BeforeValidator(_coerce_model)] = Field(
        default_factory=BankDetails
    )
    e_invoice: Annotated[EInvoice, BeforeValidator(_coerce_model)] = Field(
        default_factory=EInvoice
    )
    eway_bill: Annotated[EwayBill, BeforeValidator(_coerce_model)] = Field(
        default_factory=EwayBill
    )
    transport: Annotated[Transport, BeforeValidator(_coerce_model)] = Field(
        default_factory=Transport
    )


def build_tool_schema() -> dict:
    """Return the JSON schema for the Anthropic 'extract_invoice' tool.

    Generated directly from the Invoice model so the tool contract can never
    drift from the validation model.
    """
    return Invoice.model_json_schema()
