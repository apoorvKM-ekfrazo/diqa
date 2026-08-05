"""
DIQA Test Harness v2
--------------------
Standalone Streamlit app for testing the diqa.py module.
Run with: streamlit run diqa_test.py

Requires OPENAI_API_KEY in environment or .env file.
"""

import os
import fitz
import streamlit as st
from dotenv import load_dotenv
from diqa import assess

load_dotenv()

st.set_page_config(
    page_title="DIQA — Document Quality Check",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Document Image Quality Assessment")
st.caption(
    "Upload a PDF or image. "
    "Tier 1 uses CG-DIQA (character gradient analysis) on CPU. "
    "Borderline cases and document type validation use GPT-4o-mini vision."
)

st.divider()

DOC_TYPE_OPTIONS = {
    "": "— select document type (optional) —",
    "passport":  "🛂  Passport — bio-data page",
    "pan":       "🪪  PAN Card",
    "visa":      "✈️  Visa — e-Visa or sticker",
    "reload":    "📋  Reload Form — Thomas Cook Travel Card",
    "transfer":  "💸  Bank Transfer — online acknowledgement",
    "ticket":    "🛫  Air Ticket — flight itinerary",
    "annexure":  "📝  Annexure — dealer declaration letter",
    "sof":       "🏦  SOF — bank account statement",
}

col_type, col_upload = st.columns([1, 2], gap="medium")

with col_type:
    doc_type_key = st.selectbox(
        "Document type",
        options=list(DOC_TYPE_OPTIONS.keys()),
        format_func=lambda k: DOC_TYPE_OPTIONS[k],
        help="If selected, the pipeline will verify the uploaded file matches this type after quality checks pass."
    )
    expected_doc_type = doc_type_key if doc_type_key else None
    if expected_doc_type:
        st.caption(f"Type check: hard reject if mismatch detected.")

with col_upload:
    uploaded = st.file_uploader(
        "Drop a PDF or image here",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
    )

if not uploaded:
    st.info("Waiting for a file upload.")
    st.stop()

filetype = "pdf" if uploaded.type == "application/pdf" else "image"
file_bytes = uploaded.read()

col_preview, col_results = st.columns([1, 2], gap="large")

# ── Preview ────────────────────────────────────────────────────────────────────
with col_preview:
    st.subheader("Preview")
    try:
        if filetype == "pdf":
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            page_count = len(doc)
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            st.image(pix.tobytes("png"), caption=f"Page 1 of {page_count}", width="stretch")
            doc.close()
        else:
            st.image(file_bytes, width="stretch")
    except Exception as e:
        st.error(f"Could not render preview: {e}")

# ── Assessment ─────────────────────────────────────────────────────────────────
with col_results:
    st.subheader("Quality Assessment")

    with st.spinner("Running quality checks…"):
        result = assess(file_bytes, filetype, expected_doc_type=expected_doc_type)

    dtc = result.get("doc_type_check")
    failed_check = result.get("failed_check")

    # ── Soft mismatch: show warning + confirm button ───────────────────────────
    if not result["pass"] and result.get("requires_review"):
        st.warning("⚠️ **Possible wrong document uploaded**")
        st.write(result["reason"])
        st.info(
            "This could be a mistake by the system — these document types look similar. "
            "Please recheck your file. If you're sure this is correct, click below to continue."
        )
        confirm_key = f"confirmed_{uploaded.name}"
        if confirm_key not in st.session_state:
            st.session_state[confirm_key] = False
        if st.button("✅ I've rechecked — this is the correct document, continue"):
            st.session_state[confirm_key] = True
            st.rerun()
        if st.session_state.get(confirm_key):
            st.success("Document accepted after user confirmation.")
        else:
            st.stop()

    # ── Borderline quality: pass but flag for human review ────────────────────
    elif result["pass"] and result.get("requires_review"):
        st.warning("⚠️ **Document passes but quality is borderline**")
        flagged = result.get("flagged_pages", [])
        if flagged:
            st.write(f"Flagged page(s): {', '.join(str(p) for p in flagged)}")
        for r in result["page_results"]:
            if r.get("flagged_for_review"):
                st.write(f"Page {r['page']}: {r['reason']}")
        st.info(
            "OCR extraction can proceed, but a human should review the output — "
            "security patterns, photo noise, or photocopy quality may reduce accuracy."
        )

    # ── Hard reject ────────────────────────────────────────────────────────────
    elif not result["pass"] and failed_check == "doc_type":
        st.error("❌ **Wrong document uploaded — hard reject**")
        st.error(result["reason"])
        st.stop()

    # ── Other failures (quality, file size) ───────────────────────────────────
    elif not result["pass"]:
        if failed_check == "file_size":
            st.error(f"❌ **{result['reason']}**")
        else:
            st.error(f"❌ **{result['reason']}**")
            if result["failed_pages"]:
                page_list = ", ".join(str(p) for p in result["failed_pages"])
                st.warning(f"Problem page(s): {page_list}")
        st.stop()

    # ── Pass ───────────────────────────────────────────────────────────────────
    else:
        st.success("✅ **Document passes all quality checks** — ready for OCR extraction.")
        if dtc and dtc.get("match"):
            st.success(f"✅ **Document type confirmed** — {dtc['reason']}")

    st.divider()
    st.markdown("**Per-page breakdown**")

    SCORE_LABELS = {
        "blur":         "Sharpness",
        "brightness":   "Brightness",
        "cgdiqa":       "CG-DIQA Score",
        "mser_regions": "MSER Regions",
    }

    for r in result["page_results"]:
        tier_badge = "🖥️ Tier 1" if r.get("tier") == 1 else "🌐 Tier 2 (API)"
        page_label = (
            f"✅ Page {r['page']} — {tier_badge}"
            if r["pass"]
            else f"❌ Page {r['page']} — {r.get('failed_check', 'unknown')} — {tier_badge}"
        )

        with st.expander(page_label, expanded=not r["pass"]):
            if not r["pass"]:
                st.error(r["reason"])

            # Show which tier decided
            if r.get("tier") == 2:
                st.info("🌐 CG-DIQA was inconclusive — GPT-4o-mini issued the final verdict.")

            st.markdown("---")

            # Score metrics
            scores = r.get("scores", {})
            if scores:
                cols = st.columns(len(scores))
                for col, (key, val) in zip(cols, scores.items()):
                    label = SCORE_LABELS.get(key, key)
                    if key == "blur":
                        col.metric(label, f"{val:.1f}", f"{val - 80:.1f} vs min 80")
                    elif key == "brightness":
                        col.metric(label, f"{val:.1f}", f"ideal 45–215")
                    elif key == "cgdiqa":
                        col.metric(label, f"{val:.1f}", f"pass ≥15 | fail <5")
                    elif key == "mser_regions":
                        col.metric(label, int(val), f"min {10} for confidence")