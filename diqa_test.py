"""
DIQA Test Harness v4
--------------------
Extraction happens at upload time per document slot.
Extract and Validate runs cross-document validation only — no re-extraction.
R-DIQA-04: rejection is field-level, not image-level.

Run with: streamlit run diqa_test.py
Requires OPENAI_API_KEY in .env file.
"""

import os
import json as _json
import streamlit as st
from dotenv import load_dotenv
from diqa import assess
import extractor

load_dotenv()

st.set_page_config(
    page_title="DIQA — Document Quality Check",
    page_icon="🔍",
    layout="wide",
)

# ── Document type definitions ──────────────────────────────────────────────────
DOC_TYPES = [
    {"id": "passport", "icon": "🛂", "label": "Passport",      "sublabel": "Bio-data page"},
    {"id": "pan",      "icon": "🪪", "label": "PAN Card",       "sublabel": "Permanent Account Number"},
    {"id": "visa",     "icon": "✈️",  "label": "Visa",           "sublabel": "e-Visa or sticker visa"},
    {"id": "reload",   "icon": "📋", "label": "Reload Form",    "sublabel": "Thomas Cook Travel Card"},
    {"id": "transfer", "icon": "💸", "label": "Bank Transfer",  "sublabel": "Online acknowledgement"},
    {"id": "ticket",   "icon": "🛫", "label": "Air Ticket",     "sublabel": "Flight itinerary"},
    {"id": "annexure", "icon": "📝", "label": "Annexure",       "sublabel": "Dealer declaration letter"},
    {"id": "sof",      "icon": "🏦", "label": "SOF",            "sublabel": "Bank account statement"},
]
DOC_LABELS = {d["id"]: d["label"] for d in DOC_TYPES}

# ── Session state init ─────────────────────────────────────────────────────────
def _blank_doc():
    return {
        "status":     "pending",
        "filename":   None,
        "file_bytes": None,
        "filetype":   None,
        "result":     None,
        "extracted":  None,
        "viable":     None,
        "confirmed":  False,
    }

if "documents" not in st.session_state:
    st.session_state.documents = {d["id"]: _blank_doc() for d in DOC_TYPES}

if "run_cross_validation" not in st.session_state:
    st.session_state.run_cross_validation = False

if "cross_validation_result" not in st.session_state:
    st.session_state.cross_validation_result = None

# ── Status display config ──────────────────────────────────────────────────────
STATUS_UI = {
    "pending":          {"icon": "🔴", "label": "Required",        "color": "#888888"},
    "passed":           {"icon": "✅", "label": "Passed",           "color": "#3b6d11"},
    "flagged":          {"icon": "⚠️", "label": "Flagged",          "color": "#854f0b"},
    "review":           {"icon": "🔶", "label": "Needs review",     "color": "#a06000"},
    "field_incomplete": {"icon": "🟡", "label": "Fields missing",   "color": "#9b6e00"},
    "wrong_slot":       {"icon": "🔀", "label": "Wrong slot",       "color": "#6b4fa0"},
    "failed":           {"icon": "❌", "label": "Failed",            "color": "#a32d2d"},
}

SCORE_LABELS = {
    "blur":         "Sharpness",
    "brightness":   "Brightness",
    "cgdiqa":       "CG-DIQA",
    "mser_regions": "MSER Regions",
}

# ── Page header ────────────────────────────────────────────────────────────────
st.title("🔍 Document Quality Assessment")
st.caption(
    "Upload all 8 required documents. Each slot checks quality, verifies document "
    "type, and extracts fields immediately. Cross-document validation runs when "
    "≥ 2 documents pass or are flagged."
)
st.divider()

# ── Bulk upload ────────────────────────────────────────────────────────────────
with st.expander("📂 Upload all documents at once", expanded=False):
    st.caption(
        "Drop all files here — the system will auto-detect each document type "
        "and assign it to the correct slot. Unrecognised types will be skipped."
    )
    bulk_files = st.file_uploader(
        "Drop all documents here",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
        accept_multiple_files=True,
        key="bulk_uploader",
        label_visibility="collapsed",
    )

    # Filename keyword → doc slot mapping (fast, no API call needed)
    _FILENAME_SLOTS = {
        "PASSPORT":  "passport",  "PASSBOOK":  "sof",
        "PAN":       "pan",       "SOF":       "sof",
        "VISA":      "visa",      "STATEMENT": "sof",
        "TICKET":    "ticket",    "ITINERARY": "ticket",
        "RELOAD":    "reload",    "FORM":      "reload",
        "TRANSFER":  "transfer",  "NEFT":      "transfer",
        "ANNEXURE":  "annexure",  "ISSUANCE":  "annexure",
        "BANK":      "transfer",
    }

    if bulk_files:
        # Guard against infinite loop: st.rerun() re-executes the script and
        # bulk_files is still populated. Track a hash of the file set and skip
        # if we already processed this exact batch.
        bulk_key = "_".join(sorted(f"{bf.name}:{bf.size}" for bf in bulk_files))
        already_done = st.session_state.get("_bulk_processed_key") == bulk_key
        if already_done:
            st.info("✅ Files already processed — see document slots below.")
        else:
            st.session_state["_bulk_processed_key"] = bulk_key

        if not already_done:
            unmatched    = []
            progress_bar = st.progress(0, text="Starting…")
            n_total      = len(bulk_files)

            for file_idx, bf in enumerate(bulk_files):
                progress_bar.progress(
                    file_idx / n_total,
                    text=f"Processing {bf.name} ({file_idx + 1}/{n_total})…"
                )
                filetype   = "pdf" if bf.type == "application/pdf" else "image"
                file_bytes = bf.read()

                # Step 1: detect slot from filename (instant, no API)
                name_upper   = bf.name.upper()
                matched_slot = None
                for keyword, slot in _FILENAME_SLOTS.items():
                    if keyword in name_upper:
                        matched_slot = slot
                        break

                # Step 2: if filename didn't match, try DIQA classifier
                if not matched_slot:
                    qa_pre   = assess(file_bytes, filetype=filetype, expected_doc_type=None)
                    detected = (qa_pre.get("doc_type_check") or {}).get("detected_type") if qa_pre else None
                    if detected == "dealer_issuance_letter":
                        matched_slot = "annexure"
                    elif detected:
                        matched_slot = detected

                # Step 3: still no match → skip
                if not matched_slot:
                    unmatched.append(bf.name)
                    continue

                # Skip if slot already has this exact file
                slot_state = st.session_state.documents[matched_slot]
                if slot_state["status"] != "pending" and slot_state["filename"] == bf.name:
                    continue

                # Step 4: run DIQA with the known doc type
                qa_result = assess(file_bytes, filetype=filetype, expected_doc_type=matched_slot)
                if not qa_result:
                    unmatched.append(bf.name)
                    continue

                overall_pass = qa_result.get("pass", False)
                flagged      = qa_result.get("requires_review", False)
                extracted    = None
                viable       = None

                # Step 5: run extraction if quality gate passes
                if overall_pass or flagged:
                    try:
                        if matched_slot == "ticket":
                            # Pass passport name hint if passport already placed
                            _pp = st.session_state.documents.get("passport", {})
                            _pp_ext = _pp.get("extracted")
                            _appl_name = None
                            if isinstance(_pp_ext, dict):
                                _mrz = _pp_ext.get("mrz", {})
                                if _mrz.get("surname") and _mrz.get("given_names"):
                                    _appl_name = f"{_mrz['surname']}/{_mrz['given_names']}"
                                elif _mrz.get("surname"):
                                    _appl_name = _mrz["surname"]
                            extracted = extractor._extract_ticket(
                                file_bytes, filetype, applicant_name=_appl_name
                            )
                        else:
                            fn = extractor._EXTRACTORS.get(matched_slot)
                            extracted = fn(file_bytes, filetype) if fn else {}
                        viable = extractor.check_minimum_viable_fields(matched_slot, extracted)
                        if viable and not viable["viable"]:
                            new_status = "field_incomplete"
                        elif flagged:
                            new_status = "flagged"
                        else:
                            new_status = "passed"
                    except Exception as _bulk_err:
                        import traceback; traceback.print_exc()
                        new_status = "flagged" if flagged else "failed"
                elif qa_result.get("failed_check") == "doc_type":
                    new_status = "wrong_slot"
                    # Populate extracted so display block can show detected type
                    extracted = {
                        "_wrong_slot":    True,
                        "_detected_type": qa_result.get("doc_type_check", {}).get("detected_type", "unknown"),
                        "_expected_type": matched_slot,
                    }
                else:
                    new_status = "failed"

                st.session_state.documents[matched_slot].update({
                    "status":    new_status,
                    "filename":  bf.name,
                    "result":    qa_result,
                    "extracted": extracted,
                    "viable":    viable,
                })

            progress_bar.progress(1.0, text="Done.")
            n_placed = len(bulk_files) - len(unmatched)
            st.success(f"✅ {n_placed} of {n_total} document(s) assigned to slots.")
            if unmatched:
                st.warning(
                    f"⚠️ Could not identify type for: {', '.join(unmatched)}. "
                    f"Upload these manually below."
                )
            st.rerun()

st.divider()

# ── Column headers ─────────────────────────────────────────────────────────────
h1, h2, h3 = st.columns([2, 1, 4], gap="small")
h1.markdown("**Document**")
h2.markdown("**Status**")
h3.markdown("**Upload**")
st.divider()

# ── Per-document rows ──────────────────────────────────────────────────────────
for doc in DOC_TYPES:
    doc_id = doc["id"]
    state  = st.session_state.documents[doc_id]
    ui     = STATUS_UI[state["status"]]
    color  = ui["color"]

    col_label, col_status, col_upload = st.columns([2, 1, 4], gap="small")

    # Label column
    with col_label:
        st.markdown(
            f"{doc['icon']} **{doc['label']}** ⭐  \n"
            f"<span style='font-size:11px;color:#888'>{doc['sublabel']}</span>",
            unsafe_allow_html=True
        )

    # Status column
    with col_status:
        st.markdown(
            f"<div style='padding-top:6px;color:{color};"
            f"font-size:13px;font-weight:500'>"
            f"{ui['icon']} {ui['label']}</div>",
            unsafe_allow_html=True
        )
        if state["filename"]:
            st.caption(state["filename"][:28])

    # Upload + processing column
    with col_upload:
        uploaded = st.file_uploader(
            f"Upload {doc['label']}",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
            key=f"uploader_{doc_id}",
            label_visibility="collapsed",
        )

        if uploaded:
            already_done = (
                state["filename"] == uploaded.name
                and state["status"] != "pending"
            )

            if not already_done:
                filetype   = "pdf" if uploaded.type == "application/pdf" else "image"
                file_bytes = uploaded.read()
                new_status = "failed"
                qa_result  = None
                extracted  = None
                viable     = None

                with st.status(
                    f"Processing {doc['label']}…", expanded=True
                ) as proc:

                    # Step 1: Quality + type check
                    proc.write("Checking image quality…")
                    qa_result = assess(
                        file_bytes, filetype, expected_doc_type=doc_id
                    )

                    if qa_result["pass"] and not qa_result.get("requires_review"):
                        # ── Quality passed cleanly ─────────────────────────────
                        proc.write("Extracting fields…")
                        try:
                            if doc_id == "ticket":
                                # Fix C: pass passport name hint if passport available
                                _pp = st.session_state.documents.get("passport", {})
                                _pp_ext = _pp.get("extracted")
                                _appl_name = None
                                if isinstance(_pp_ext, dict):
                                    _mrz = _pp_ext.get("mrz", {})
                                    if _mrz.get("surname") and _mrz.get("given_names"):
                                        _appl_name = f"{_mrz['surname']}/{_mrz['given_names']}"
                                    elif _mrz.get("surname"):
                                        _appl_name = _mrz["surname"]
                                extracted = extractor._extract_ticket(
                                    file_bytes, filetype, applicant_name=_appl_name
                                )
                            else:
                                fn = extractor._EXTRACTORS.get(doc_id)
                                extracted = fn(file_bytes, filetype) if fn else {}
                        except Exception as e:
                            extracted = {"source": doc_id, "error": str(e)}

                        proc.write("Checking required fields…")
                        viable = extractor.check_minimum_viable_fields(
                            doc_id, extracted
                        )

                        if not viable["viable"]:
                            new_status = "field_incomplete"
                            proc.update(
                                label="🟡 Some required fields could not be read",
                                state="complete", expanded=False
                            )
                        else:
                            new_status = "passed"
                            proc.update(
                                label="✅ Passed",
                                state="complete", expanded=False
                            )

                    elif qa_result["pass"] and qa_result.get("requires_review"):
                        # ── Borderline quality — flagged ───────────────────────
                        proc.write("Extracting fields…")
                        _extr_err = None
                        try:
                            if doc_id == "ticket":
                                # Fix C: pass passport name hint if passport
                                # already uploaded — helps locate correct passenger row
                                _pp = st.session_state.documents.get("passport", {})
                                _pp_ext = _pp.get("extracted")
                                _appl_name = None
                                if isinstance(_pp_ext, dict):
                                    _mrz = _pp_ext.get("mrz", {})
                                    if _mrz.get("surname") and _mrz.get("given_names"):
                                        _appl_name = f"{_mrz['surname']}/{_mrz['given_names']}"
                                    elif _mrz.get("surname"):
                                        _appl_name = _mrz["surname"]
                                extracted = extractor._extract_ticket(
                                    file_bytes, filetype, applicant_name=_appl_name
                                )
                            else:
                                fn = extractor._EXTRACTORS.get(doc_id)
                                extracted = fn(file_bytes, filetype) if fn else {}
                        except Exception as e:
                            extracted = {"source": doc_id, "error": str(e)}
                            _extr_err = str(e)
                        viable = extractor.check_minimum_viable_fields(
                            doc_id, extracted
                        )
                        new_status = "flagged"
                        if _extr_err:
                            proc.update(
                                label="⚠️ Flagged — extraction failed internally",
                                state="complete", expanded=False
                            )
                        else:
                            proc.update(
                                label="⚠️ Passed with review flag",
                                state="complete", expanded=False
                            )

                    elif not qa_result["pass"] and qa_result.get("requires_review"):
                        # ── Soft doc type mismatch — extract immediately, let agent confirm ─
                        # Data is extracted now so cross-validation can use it
                        # whether or not the agent clicks the confirm button.
                        detected_type = (
                            qa_result.get("doc_type_check", {}).get("detected_type")
                            or doc_id
                        )
                        fn_review = extractor._EXTRACTORS.get(detected_type)
                        if fn_review:
                            try:
                                raw_review = fn_review(file_bytes, filetype)
                                extracted = {
                                    **raw_review,
                                    "_wrong_slot":    True,
                                    "_detected_type": detected_type,
                                    "_expected_type": doc_id,
                                }
                            except Exception as e:
                                extracted = {"source": detected_type, "error": str(e)}
                        else:
                            extracted = {}
                        new_status = "review"
                        proc.update(
                            label="⚠️ Please review document type",
                            state="complete", expanded=False
                        )

                    elif (
                        not qa_result["pass"]
                        and qa_result.get("failed_check") == "doc_type"
                    ):
                        # ── Doc type check failed ──────────────────────────────
                        detected_type = (
                            qa_result.get("doc_type_check", {}).get("detected_type")
                            or doc_id
                        )

                        if detected_type == doc_id or detected_type in (None, "unknown"):
                            # Same document type, wrong variant
                            # (e.g. company PAN in individual PAN slot)
                            # Hard reject — clear message, no extraction
                            new_status = "failed"
                            proc.update(
                                label="❌ Wrong document variant",
                                state="error", expanded=False
                            )
                        else:
                            # R-DOC-MISMATCH-01: genuinely different doc type
                            # Extract using detected type — data not wasted
                            proc.write(
                                f"Wrong slot — extracting as {detected_type}…"
                            )
                            fn = extractor._EXTRACTORS.get(detected_type)
                            if fn:
                                try:
                                    raw = fn(file_bytes, filetype)
                                    extracted = {
                                        **raw,
                                        "_wrong_slot":    True,
                                        "_detected_type": detected_type,
                                        "_expected_type": doc_id,
                                    }
                                except Exception as e:
                                    extracted = {
                                        "source": detected_type, "error": str(e)
                                    }
                            new_status = "wrong_slot"
                            viable = {
                                "viable": False, "missing_fields": [],
                                "messages": [
                                    "Wrong slot — re-upload correct document type"
                                ]
                            }
                            proc.update(
                                label=f"🔀 Wrong slot — extracted as {detected_type}",
                                state="complete", expanded=False
                            )

                    else:
                        # ── Quality failed ─────────────────────────────────────
                        new_status = "failed"
                        proc.update(
                            label="❌ Failed quality check",
                            state="error", expanded=False
                        )

                st.session_state.documents[doc_id].update({
                    "status":     new_status,
                    "filename":   uploaded.name,
                    "file_bytes": file_bytes,
                    "filetype":   filetype,
                    "result":     qa_result,
                    "extracted":  extracted,
                    "viable":     viable,
                    "confirmed":  False,
                })
                st.session_state.cross_validation_result = None
                st.rerun()

        # ── Details expander ───────────────────────────────────────────────────
        result = state["result"]
        status = state["status"]

        # Anchor field confidence check — shown for all statuses that have extraction
        if status in ("passed", "flagged") and state.get("extracted"):
            from diqa import check_extraction_confidence
            anchor_warnings = check_extraction_confidence(
                doc_id, state["extracted"]
            )
            for aw in anchor_warnings:
                st.warning(
                    f"⚠️ **Extraction confidence low — {aw['field']}**: "
                    f"extracted value `{aw['value']}` does not match expected format. "
                    f"{aw['hint']}"
                )

        if result and status in (
            "failed", "flagged", "review", "field_incomplete", "wrong_slot"
        ):
            with st.expander("See details", expanded=False):

                if status == "failed":
                    fc = result.get("failed_check")
                    if fc == "file_size":
                        st.error(result.get("reason", "File too large."))
                    elif fc == "doc_type":
                        st.error(result.get("reason", "Wrong document type."))
                    else:
                        st.error(result.get("reason", "Quality check failed."))
                        for r in result.get("page_results", []):
                            if not r.get("pass"):
                                st.markdown(
                                    f"**Page {r['page']}:** {r['reason']}"
                                )
                                scores = r.get("scores", {})
                                if scores:
                                    mcols = st.columns(len(scores))
                                    for mc, (k, v) in zip(mcols, scores.items()):
                                        mc.metric(
                                            SCORE_LABELS.get(k, k),
                                            f"{v:.1f}" if isinstance(v, float)
                                            else str(v)
                                        )

                elif status == "flagged":
                    st.warning(result.get("reason", "Borderline quality."))
                    for r in result.get("page_results", []):
                        if r.get("flagged_for_review"):
                            st.markdown(f"**Page {r['page']}:** {r['reason']}")
                    _ext = state.get("extracted") or {}
                    _ext_err = _ext.get("error") if isinstance(_ext, dict) else None
                    if _ext_err:
                        st.error(
                            f"⚠️ Extraction failed: `{_ext_err}`\n\n"
                            "Check server logs for the root cause. "
                            "Re-upload or investigate the extractor."
                        )
                    else:
                        viable = state.get("viable")
                        if viable and viable["viable"]:
                            st.success("All required fields extracted successfully.")
                        elif viable and not viable["viable"]:
                            st.warning("Some required fields could not be extracted.")
                            for msg in viable.get("messages", []):
                                st.write(f"• {msg}")
                    st.info(
                        "Document is usable — recommend human review of "
                        "extracted output."
                    )

                elif status == "review":
                    st.warning(
                        result.get("reason", "Possible document type mismatch.")
                    )
                    st.write(
                        "These document types look similar — the system may have "
                        "made a mistake. Please recheck the file before confirming."
                    )
                    if not state.get("confirmed"):
                        if st.button(
                            "✅ I've rechecked — this is correct, accept it",
                            key=f"confirm_{doc_id}"
                        ):
                            st.session_state.documents[doc_id]["status"]    = "passed"
                            st.session_state.documents[doc_id]["confirmed"] = True
                            st.session_state.cross_validation_result = None
                            st.rerun()
                    else:
                        st.success("Accepted after agent confirmation.")

                elif status == "field_incomplete":
                    viable = state.get("viable", {})
                    st.warning(
                        "Some required fields could not be extracted from "
                        "this document."
                    )
                    for msg in viable.get("messages", []):
                        st.write(f"• {msg}")
                    extracted_data = state.get("extracted", {})
                    if extracted_data and not extracted_data.get("error"):
                        st.markdown("**Fields successfully extracted:**")
                        mvf = extractor.MINIMUM_VIABLE_FIELDS.get(doc_id, [])
                        src = (
                            extracted_data.get("mrz", extracted_data)
                            if doc_id == "passport"
                            else extracted_data
                        )
                        got = {
                            k: v for k, v in src.items()
                            if k in mvf
                            and v
                            and str(v).strip() not in ("", "null", "—")
                        }
                        for k, v in got.items():
                            st.write(f"  ✅ {k}: `{v}`")
                    st.info(
                        "Re-upload a clearer version, or proceed to "
                        "cross-validation with the fields that were extracted."
                    )

                elif status == "wrong_slot":
                    detected = (
                        (state.get("extracted") or {}).get("_detected_type")
                        or state.get("result", {}).get("doc_type_check", {}).get("detected_type")
                        or "unknown"
                    )
                    st.warning(
                        f"This appears to be a **{detected.upper()}** document, "
                        f"not a **{DOC_LABELS.get(doc_id, doc_id)}**. "
                        f"Please re-upload the correct document."
                    )
                    st.info(
                        "Fields extracted from this document are available as "
                        "supporting evidence in cross-validation but do not "
                        "satisfy this slot requirement."
                    )
                    extracted_data = state.get("extracted", {})
                    if extracted_data and not extracted_data.get("error"):
                        with st.expander(
                            "View extracted fields (supporting evidence only)"
                        ):
                            clean = {
                                k: v for k, v in extracted_data.items()
                                if not k.startswith("_")
                            }
                            st.json(clean)

    st.divider()

# ── Summary bar + Cross-Validation button ─────────────────────────────────────
docs      = st.session_state.documents
n_passed  = sum(1 for s in docs.values() if s["status"] == "passed")
n_flagged = sum(1 for s in docs.values() if s["status"] == "flagged")
n_review  = sum(1 for s in docs.values() if s["status"] == "review")
n_fi      = sum(1 for s in docs.values() if s["status"] == "field_incomplete")
n_failed  = sum(1 for s in docs.values() if s["status"] == "failed")
n_pending = sum(1 for s in docs.values() if s["status"] == "pending")
n_wrong   = sum(1 for s in docs.values() if s["status"] == "wrong_slot")

# PASSED + FLAGGED both count toward the gate.
# FLAGGED participates in cross-validation with CONFIDENCE: LOW.
# Results from FLAGGED docs show with confidence caveats — no hard alerts.
n_ok = n_passed + n_flagged

parts = []
if n_passed:  parts.append(f"✅ {n_passed} passed")
if n_flagged: parts.append(f"⚠️ {n_flagged} flagged")
if n_fi:      parts.append(f"🟡 {n_fi} fields missing")
if n_wrong:   parts.append(f"🔀 {n_wrong} wrong slot")
if n_review:  parts.append(f"🔶 {n_review} needs review")
if n_failed:  parts.append(f"❌ {n_failed} failed")
if n_pending: parts.append(f"🔴 {n_pending} pending")

bcol1, bcol2 = st.columns([3, 1])

with bcol1:
    st.markdown(
        f"**{n_ok} of 8 ready for cross-validation** — "
        + " · ".join(parts)
        if parts else "No documents uploaded yet."
    )

with bcol2:
    if st.button(
        "Cross-Validate ▶",
        disabled=(n_ok < 2),
        type="primary",
        help="Requires at least 2 passed or flagged documents",
    ):
        st.session_state.run_cross_validation = True
        st.session_state.cross_validation_result = None

if n_ok < 2:
    st.caption(
        f"Cross-Validate needs at least 2 passed or flagged documents — "
        f"{n_ok} ready."
    )

# ── Cross-document validation results ─────────────────────────────────────────
if st.session_state.run_cross_validation:
    import pandas as pd
    st.divider()
    st.subheader("Cross-Document Validation")

    # PASSED documents eligible for cross-validation
    # FLAGGED included with confidence caveat
    # WRONG_SLOT included as supporting evidence only
    usable_extracted = {}
    for doc_id, state in docs.items():
        if state["status"] in ("passed", "flagged") and state.get("extracted"):
            usable_extracted[doc_id] = state["extracted"]
        elif state["status"] in ("wrong_slot", "review") and state.get("extracted"):
            detected = state["extracted"].get("_detected_type", doc_id)
            usable_extracted[f"_supporting_{detected}"] = state["extracted"]

    if not st.session_state.cross_validation_result:
        with st.spinner("Running cross-document validation…"):
            st.session_state.cross_validation_result = (
                extractor.run_cross_validation(usable_extracted)
            )

    result = st.session_state.cross_validation_result

    # Show confidence warning if any FLAGGED docs contributed
    flagged_docs = [
        DOC_LABELS.get(did, did)
        for did, s in docs.items()
        if s["status"] == "flagged" and s.get("extracted")
    ]
    if flagged_docs:
        st.warning(
            f"⚠️ **Confidence: LOW** — results include data from flagged "
            f"document(s): {', '.join(flagged_docs)}. "
            f"Do not use for automated KYC decisions — human review required."
        )

    # ── Matrix field table — generic field names as rows, doc types as columns ──
    # Column already tells you the doc type, so field names are generic:
    # "Name" row shows passport name, PAN name, ticket passenger, reload cardholder
    # side by side — mismatches are immediately visible.
    # ❌ = field applies to this doc but was not extracted
    # —  = field does not apply to this doc type at all
    st.markdown("#### Extracted Fields")
    field_rows = result.get("field_table", [])

    # Display field map:
    # (display_name, category, {doc_id: [raw field names to look up]}, required_doc_set)
    # Multiple raw fields in a list are joined (e.g. Surname + Given Names → full name).
    # required_doc_set: show ❌ if doc is active and field is missing.
    _DISPLAY_FIELD_MAP = [
        # ── Identity ────────────────────────────────────────────────────────────
        ("Name", "Identity", {
            "passport": ["Surname", "Given Names"],
            "pan":      ["PAN Name"],
            "ticket":   ["Passenger Name"],
            "reload":   ["Reload Name"],
            "annexure": ["Cardholder Name"],
            "sof":      ["Account Holder"],
        }, {"passport", "pan", "ticket"}),

        ("Date of Birth", "Identity", {
            "passport": ["Date of Birth"],
            "pan":      ["PAN DOB"],
            "reload":   ["Reload DOB"],
        }, {"passport", "pan"}),

        ("ID Number", "Identity", {
            "passport": ["Passport No."],
            "pan":      ["PAN Number"],
            "visa":     ["Visa Number"],
            "reload":   ["Reload Passport No."],
        }, {"passport", "pan"}),

        ("Expiry Date", "Identity", {
            "passport": ["Expiry Date"],
            "visa":     ["Visa Expiry", "Expiry Date"],
        }, {"passport"}),

        ("Nationality", "Identity", {
            "passport": ["Nationality"],
        }, {}),

        ("Issue Date", "Identity", {
            "passport": ["Issue Date"],
            "visa":     ["Visa Issue Date", "Issue Date"],
        }, {}),

        ("Place of Issue", "Identity", {
            "passport": ["Place of Issue"],
        }, {}),

        ("Father Name", "Identity", {
            "pan":      ["Father Name"],
        }, {}),

        ("Entity Type", "Identity", {
            "pan":      ["Entity Type"],
        }, {}),

        ("MRZ Verified", "Identity", {
            "passport": ["MRZ Verified"],
        }, {"passport"}),

        # ── Travel ──────────────────────────────────────────────────────────────
        ("Destination", "Travel", {
            "visa":     ["Destination"],
            "ticket":   ["Destination Country", "Destination IATA"],
        }, {"ticket"}),

        ("Origin", "Travel", {
            "ticket":   ["Origin IATA"],
        }, {"ticket"}),

        ("PNR", "Travel", {
            "ticket":   ["PNR"],
        }, {"ticket"}),

        ("Airline", "Travel", {
            "ticket":   ["Airline"],
        }, {}),

        ("Departure Date", "Travel", {
            "ticket":   ["Departure Date"],
        }, {"ticket"}),

        ("Return Date", "Travel", {
            "ticket":   ["Return Date"],
        }, {}),

        ("Visa Type", "Travel", {
            "visa":     ["Visa Type"],
        }, {}),

        ("Issuing Authority", "Travel", {
            "visa":     ["Issuing Authority"],
        }, {}),

        ("Flight Legs", "Travel", {
            "ticket":   ["Legs Parsed"],
        }, {}),

        # ── Financial ───────────────────────────────────────────────────────────
        ("Currency", "Financial", {
            "reload":   ["Currencies", "currencies_requested"],
            "annexure": ["Annexure Currency"],
        }, {"reload"}),

        ("Amount", "Financial", {
            "transfer": ["Transfer Amount"],
            "annexure": ["Annexure Amount"],
        }, {}),

        ("Purpose of Travel", "Financial", {
            "reload":   ["Purpose of Travel", "purpose_of_travel"],
        }, {}),

        ("Card Number", "Financial", {
            "reload":   ["Card Number", "card_number"],
        }, {}),

        ("Card Network", "Financial", {
            "reload":   ["Card Network", "card_network"],
        }, {}),

        ("UTR / Reference", "Financial", {
            "transfer": ["Reference / UTR"],
            "annexure": ["Annexure UTR", "Reference / UTR"],
        }, {}),

        ("Transaction Date", "Financial", {
            "reload":   ["Form Date"],
            "transfer": ["Transfer Date"],
        }, {}),

        ("Beneficiary", "Financial", {
            "transfer": ["Beneficiary"],
        }, {}),

        ("Card Type", "Financial", {
            "reload":   ["Card Type"],
        }, {}),

        ("Dealer Name", "Financial", {
            "annexure": ["Dealer Name"],
        }, {}),

        ("RBI Licence", "Financial", {
            "annexure": ["RBI Licence"],
        }, {}),

        ("Bank", "Financial", {
            "sof":      ["Bank"],
        }, {}),

        ("Statement Period", "Financial", {
            "sof":      ["Statement Period"],
        }, {}),

        ("Closing Balance", "Financial", {
            "sof":      ["Closing Balance"],
        }, {}),
    ]

    _DOC_ORDER = ["passport", "pan", "visa", "ticket", "reload", "transfer", "annexure", "sof"]
    _DOC_LABEL = {
        "passport": "🛂 Passport", "pan": "🪪 PAN",
        "visa": "✈️ Visa",        "ticket": "🛫 Ticket",
        "reload": "📋 Reload",    "transfer": "💸 Transfer",
        "annexure": "📝 Annexure","sof": "🏦 SOF",
    }
    _CAT_COLORS = {
        "Identity":  "#eef3fb",
        "Travel":    "#edfbf3",
        "Financial": "#fdf7ee",
    }

    if field_rows:
        # Build lookup: (raw_field_name, source_lower) → value
        _val_lookup = {}
        for row in field_rows:
            key = (row["Field"], row["Source"].lower())
            _val_lookup[key] = row["Value"]

        # Only show doc columns that have at least one extracted value
        active_docs = [
            d for d in _DOC_ORDER
            if any(src == d for _, src in _val_lookup)
        ] or _DOC_ORDER

        # Build matrix rows
        matrix_rows = []
        for display_name, category, doc_field_map, required_docs in _DISPLAY_FIELD_MAP:
            row_data = {"Category": category, "Field": display_name}
            any_cell = False

            for doc_id in active_docs:
                label      = _DOC_LABEL[doc_id]
                raw_fields = doc_field_map.get(doc_id, [])

                # Collect non-empty values for all mapped raw fields
                vals = [
                    v for rf in raw_fields
                    for v in [_val_lookup.get((rf, doc_id))]
                    if v and v not in ("—", "", None)
                ]

                if vals:
                    row_data[label] = " ".join(vals)   # joins Surname + Given Names
                    any_cell = True
                elif doc_id in required_docs and doc_id in doc_field_map:
                    row_data[label] = "❌"              # should have it, doesn't
                    any_cell = True
                elif doc_id in doc_field_map:
                    row_data[label] = "—"              # optional / not extracted
                else:
                    row_data[label] = "—"              # not applicable to this doc

            if any_cell:
                matrix_rows.append(row_data)

        if matrix_rows:
            doc_cols     = [_DOC_LABEL[d] for d in active_docs]
            display_cols = ["Category", "Field"] + doc_cols
            matrix_df    = pd.DataFrame(matrix_rows)[display_cols]

            # Reload form identity fields are handwritten — lower OCR confidence
            _RELOAD_COL           = _DOC_LABEL.get("reload", "📋 Reload")
            _RELOAD_HANDWRITTEN   = {"Name", "Date of Birth", "ID Number"}

            def _style_matrix(row):
                bg     = _CAT_COLORS.get(row["Category"], "#ffffff")
                styles = [f"background-color: {bg}; font-size:12px"] * len(row)
                for i, col in enumerate(row.index):
                    if col in ("Category", "Field"):
                        continue
                    val = row[col]
                    if val == "❌":
                        styles[i] = (
                            "background-color: #fde8e8; color: #a32d2d;"
                            " font-size:12px; font-weight:600"
                        )
                    elif (
                        col == _RELOAD_COL
                        and row["Field"] in _RELOAD_HANDWRITTEN
                        and val not in ("—", "", None)
                    ):
                        # Amber = extracted but from handwritten OCR, treat as approximate
                        styles[i] = (
                            "background-color: #fef9e7; color: #7d5a00;"
                            " font-size:12px; font-style:italic"
                        )
                    elif val not in ("—", "", None):
                        styles[i] = f"background-color: {bg}; color: #1a5c1a; font-size:12px"
                return styles

            st.dataframe(
                matrix_df.style.apply(_style_matrix, axis=1).hide(axis="index"),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "🟡 **Amber cells (Reload form)** — Name, Date of Birth, and ID Number "
                "under 📋 Reload are extracted from handwritten grid boxes using GPT-4o mini. "
                "Character-level OCR on handwritten forms is approximate: digits and similar "
                "characters (0/9, 1/7, 5/6) are commonly confused. "
                "These fields are shown for reference only — mismatches here do **not** block "
                "approval. Verify against the original document if needed."
            )
        else:
            st.info("No fields available for display.")
    else:
        st.info("No fields available for display.")

    # ── Cross-validation checks ────────────────────────────────────────────────
    st.markdown("#### Validation Checks")
    checks = result.get("cross_validation", [])
    if checks:
        # ── Summary verdict ────────────────────────────────────────────────────
        n_fail = sum(1 for c in checks if c["result"] == "fail")
        n_warn = sum(1 for c in checks if c["result"] == "warning")
        n_pass = sum(1 for c in checks if c["result"] == "pass")

        if n_fail > 0:
            st.error(
                f"❌ **Application blocked** — {n_fail} critical issue(s) found. "
                f"Resolve before processing."
            )
        elif n_warn > 0:
            st.warning(
                f"⚠️ **Application requires review** — {n_warn} warning(s) need "
                f"human verification. {n_pass} check(s) passed."
            )
        else:
            st.success(
                f"✅ **Application ready for approval** — all {n_pass} checks passed."
            )

        st.divider()

        # ── Individual checks ──────────────────────────────────────────────────
        for chk in checks:
            if chk["result"] == "pass":
                st.success(f"✅ **{chk['check']}** — {chk['detail']}")
            elif chk["result"] == "warning":
                st.warning(f"⚠️ **{chk['check']}** — {chk['detail']}")
            else:
                st.error(f"❌ **{chk['check']}** — {chk['detail']}")
    else:
        st.info(
            "No cross-checks available — upload passport + PAN for name "
            "matching, reload + ticket for currency-destination check."
        )

    # ── Excluded documents ─────────────────────────────────────────────────────
    incomplete = {
        did: s for did, s in docs.items()
        if s["status"] == "field_incomplete"
    }
    if incomplete:
        st.markdown("#### ⚠️ Documents Excluded — Missing Required Fields")
        st.caption(
            "These documents passed quality checks but could not provide all "
            "required fields. Re-upload to include them in cross-validation."
        )
        for doc_id, state in incomplete.items():
            viable = state.get("viable", {})
            missing = viable.get("missing_fields", [])
            st.write(
                f"**{DOC_LABELS.get(doc_id, doc_id)}** — "
                f"missing: {', '.join(missing)}"
            )

    # ── Reset ──────────────────────────────────────────────────────────────────
    st.divider()
    if st.button("🔄 Re-run Cross-Validation"):
        st.session_state.cross_validation_result = None
        st.rerun()