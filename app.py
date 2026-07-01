"""Streamlit demo UI for the invoice mapper.

This file is presentation only: it uploads a PDF, calls pipeline.process_pdf,
shows the results and offers a download. All extraction/mapping logic lives in
core/ and pipeline.py, so replacing this UI later touches nothing else.
"""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

from config import DEFAULT_TIER, MODEL_TIERS, load_settings
from pipeline import PipelineError, process_pdf, result_filename, to_download

# Client-facing tier labels (model names stay hidden) -> tier keys in MODEL_TIERS.
# Order sets the dropdown order; the default selection is driven by DEFAULT_TIER.
TIER_OPTIONS = {
    "Low · fastest & cheapest": "low",
    "Medium · balanced": "medium",
    "High · most accurate (complex PDFs)": "high",
}

st.set_page_config(page_title="Invoice Mapper", page_icon="🧾", layout="centered")
st.title("🧾 Invoice Mapper")
st.caption("Upload a text-based invoice PDF and get clean, standardised JSON.")

settings = load_settings()
if not settings.is_api_key_present:
    st.warning("API key not found. Please contact your administrator.", icon="⚠️")

# A bumpable key lets "New extraction" fully reset the uploader widget.
st.session_state.setdefault("uploader_key", 0)
uploaded = st.file_uploader(
    "Choose an invoice PDF", type=["pdf"], key=f"pdf_{st.session_state.uploader_key}"
)

# Processing power (left) + Extract button (right), aligned on one row.
st.markdown("**Processing power**")
power_col, button_col = st.columns([3, 1])
tier_label = power_col.selectbox(
    "Processing power",
    options=list(TIER_OPTIONS),
    index=list(TIER_OPTIONS.values()).index(DEFAULT_TIER),
    label_visibility="collapsed",
    help=(
        "Higher tiers are more accurate but cost more. Use High for complex or "
        "financially critical invoices; Low is fine for simple, clean ones."
    ),
)
chosen_model = MODEL_TIERS[TIER_OPTIONS[tier_label]]
extract_clicked = button_col.button(
    "Extract to JSON", type="primary", disabled=uploaded is None
)

if uploaded is not None and extract_clicked:
    settings = replace(settings, model=chosen_model)  # apply the chosen tier
    progress_bar = st.progress(0.0, text="Starting…")

    def on_progress(done: int, total: int) -> None:
        progress_bar.progress(done / total, text=f"Mapped {done} of {total} invoice(s)…")

    try:
        with st.spinner("Reading and mapping…"):
            result = process_pdf(uploaded.getvalue(), uploaded.name, settings, on_progress)
    except PipelineError as exc:
        progress_bar.empty()
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # surface unexpected failures instead of a blank screen
        progress_bar.empty()
        st.error(f"Unexpected error while processing: {exc}")
        st.stop()

    progress_bar.empty()
    # Persist so the download button (which reruns the script) keeps the data.
    st.session_state["results"] = result.invoices
    st.session_state["usage"] = result.usage

results = st.session_state.get("results")
if results:
    top_left, top_right = st.columns([3, 1])
    top_left.success(f"Extracted {len(results)} invoice(s).")
    if top_right.button("🔄 New extraction"):
        for key in ("results", "usage"):
            st.session_state.pop(key, None)
        st.session_state.uploader_key += 1  # forces a fresh, empty uploader
        st.rerun()

    usage = st.session_state.get("usage")
    if usage:
        with st.expander("Cost & usage details"):
            if usage.cost_inr is not None:
                st.markdown(f"**Estimated cost:** ₹{usage.cost_inr:,.2f}  (≈ ${usage.cost_usd:.4f})")
            else:
                st.markdown("**Estimated cost:** n/a")
            cached_note = (
                f"  ({usage.cache_read_tokens:,} cached)" if usage.cache_read_tokens else ""
            )
            st.markdown(f"**Input tokens:** {usage.total_input_tokens:,}{cached_note}")
            st.markdown(f"**Output tokens:** {usage.output_tokens:,}")
            st.markdown(f"**Model:** {usage.model}")

    warned = [r for r in results if r.get("_warnings")]
    if warned:
        with st.expander(f"⚠️ {len(warned)} invoice(s) have validation notes"):
            for record in warned:
                st.markdown(f"**{record.get('invoice_no') or 'Unknown'}**")
                for note in record["_warnings"]:
                    st.markdown(f"- {note}")

    labels = [
        f"{i}. {r.get('invoice_no') or result_filename(r, i)}"
        for i, r in enumerate(results, start=1)
    ]
    chosen = st.selectbox("Preview an invoice", labels, index=0)
    st.json(results[labels.index(chosen)])

    data, filename, mime = to_download(results)
    st.download_button(
        "⬇️ Download JSON" + (" (ZIP)" if filename.endswith(".zip") else ""),
        data=data,
        file_name=filename,
        mime=mime,
    )
