"""
extractor.py — Document field extraction and cross-document validation
----------------------------------------------------------------------
v3.0 — Complete rewrite incorporating all rulebook fixes (R-PAN-01 through
R-TICKET-07, R-VISA-01 through R-VISA-04, R-DIQA-04, R-XVAL-01 through
R-XVAL-07).

Architecture:
  Passport : PassportEye (MRZ checksum-validated) + GPT-4o-mini (VIZ fields)
  Visa     : Tesseract (MRZ expiry) + GPT-4o-mini (VIZ fields)
  Ticket   : GPT-4o-mini (city-name legs + all_passengers) + Python rules
  All others: GPT-4o-mini structured JSON extraction
  Cross-validation: deterministic Python — RapidFuzz name match, visa window
                    year inference, all-dates coverage check

Entry point (batch): run(usable_docs, progress_callback)
Entry point (incremental): run_cross_validation(extracted_by_doc)
"""

import os
import re
import base64
import json
import tempfile
from datetime import date, timedelta
from calendar import month_abbr

import cv2
import fitz
import numpy as np
import pytesseract
from openai import OpenAI
from rapidfuzz import distance as rfdist


# ── OpenAI client (lazy) ───────────────────────────────────────────────────────
_client = None

def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


# ── Image helpers ──────────────────────────────────────────────────────────────

def _pdf_to_images(file_bytes: bytes) -> list:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    mat = fitz.Matrix(2.0, 2.0)   # 2x zoom for better OCR
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        arr = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
    doc.close()
    return images


def _to_images(file_bytes: bytes, filetype: str) -> list:
    if filetype == "pdf":
        return _pdf_to_images(file_bytes)
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return [img] if img is not None else []


def _encode_images(images: list, max_pages: int = 3) -> list:
    """Encode up to max_pages images as base64 image_url content blocks."""
    content = []
    for img in images[:max_pages]:
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64 = base64.b64encode(buf).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "high"
            }
        })
    return content


def _validate_mrz_checksum(value: str, check_digit: int) -> bool:
    """
    ICAO 9303 check digit algorithm.
    Weights cycle: 7, 3, 1. A-Z = 10-35, 0-9 = 0-9, < = 0.
    """
    weights   = [7, 3, 1]
    char_vals = {str(i): i for i in range(10)}
    char_vals.update({chr(65 + i): 10 + i for i in range(26)})
    char_vals["<"] = 0
    total = sum(char_vals.get(c, 0) * weights[i % 3] for i, c in enumerate(value))
    return (total % 10) == check_digit


def _correct_rotation(img: np.ndarray) -> np.ndarray:
    """
    Detect and correct image rotation using Tesseract OSD.
    FastMRZ requires MRZ lines to be horizontal — a 90° rotated passport
    causes complete MRZ read failure (R-MRZ-01).

    Tesseract OSD returns the angle the image IS rotated FROM upright.
    We rotate in the opposite direction to correct it.
    """
    try:
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        osd     = pytesseract.image_to_osd(pil_img, config="--psm 0")
        match   = re.search(r"Rotate: (\d+)", osd)
        if not match:
            return img
        angle = int(match.group(1))
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    except Exception:
        pass   # OSD failure — proceed with original, better than crashing
    return img


# ── GPT extraction helper ──────────────────────────────────────────────────────

def _gpt_extract(
    images: list,
    prompt: str,
    system_prompt: str = None,
    max_tokens: int = 600,
) -> dict:
    """
    Send images to GPT-4o-mini with a JSON extraction prompt.
    Optional system_prompt sets model behaviour before the user message.
    Returns parsed dict, or {"error": "..."} on any failure.
    max_tokens: increase for prompts that return large JSON (e.g. all_passengers).
    """
    try:
        content = _encode_images(images)
        content.append({"type": "text", "text": prompt})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        response = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
        )
        raw   = response.choices[0].message.content.strip()
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception as e:
        return {"error": str(e)}


# ── Per-document extractors ────────────────────────────────────────────────────

def _extract_passport(file_bytes: bytes, filetype: str) -> dict:
    """
    Three-step passport extraction:

    Step 1 — Rotation correction (R-MRZ-01):
        Detect and correct image orientation before FastMRZ.
        A 90° rotated passport causes complete MRZ read failure.

    Step 2 — PassportEye:
        Cryptographically validated MRZ fields — surname, given_names,
        passport_number, date_of_birth, expiry_date, nationality, checksum.

    Step 3 — GPT-4o-mini VIZ pass:
        Always extracts issue_date and place_of_issue from printed VIZ data.
        Also acts as fallback for any MRZ field PassportEye returned null.
    """
    images = _to_images(file_bytes, filetype)
    if not images:
        return {"source": "passport", "error": "Could not decode image"}

    mrz_data = {}
    viz_data = {}

    # ── Step 1: Rotation correction ───────────────────────────────────────────
    corrected_images = [_correct_rotation(img) for img in images]

    # ── Step 2: PassportEye MRZ read ─────────────────────────────────────────
    # PassportEye locates and decodes the MRZ from the image, validates all
    # check digits, and returns a structured object. More reliable than FastMRZ
    # on Windows and works directly with JPEG images.
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress scikit-image FutureWarnings
            from passporteye import read_mrz

        # Use PNG (lossless) — JPEG compression degrades MRZ text enough
        # to prevent PassportEye from locating the MRZ zone.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(cv2.imencode(".png", corrected_images[0])[1].tobytes())
            tmp_path = tmp.name

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrz_result = read_mrz(tmp_path)
        os.unlink(tmp_path)

        def _pe_date(v):
            """YYMMDD → YYYY-MM-DD (PassportEye returns dates in YYMMDD format)."""
            v = str(v or "").strip()
            if len(v) == 6 and v.isdigit():
                yy, mm, dd = v[:2], v[2:4], v[4:]
                century = "20" if int(yy) < 30 else "19"
                return f"{century}{yy}-{mm}-{dd}"
            return v

        # Score threshold: below 50 PassportEye found something MRZ-like
        # but the read is too garbled to be reliable (surnames like
        # "VGEREEENTSOSESORSIT" at score 2). Treat low-score reads as
        # unreadable — same as None.
        _pe_score = getattr(mrz_result, "valid_score", 0) or 0 if mrz_result else 0
        _pe_usable = mrz_result is not None and _pe_score >= 80

        if _pe_usable:
            # Use to_dict() — avoids attribute name differences across
            # PassportEye versions and MRZ document types (TD1/TD2/TD3).
            try:
                _d = mrz_result.to_dict()
            except Exception:
                _d = {}

            mrz_data = {
                "surname":         str(_d.get("surname") or "").strip(),
                "given_names":     str(_d.get("given_names") or _d.get("names") or "").strip(),
                "passport_number": str(_d.get("number") or _d.get("document_number") or "").rstrip("<").strip(),
                "nationality":     str(_d.get("nationality") or _d.get("country") or "").strip(),
                "date_of_birth":   _pe_date(_d.get("date_of_birth") or _d.get("dob") or _d.get("birth_date") or ""),
                "expiry_date":     _pe_date(_d.get("expiration_date") or _d.get("expiry_date") or _d.get("expiry") or ""),
                "sex":             str(_d.get("sex") or "").strip(),
                "checksum_valid":  bool(getattr(mrz_result, "valid", False)),
                "valid_score":     _pe_score,
                "_source":         "PassportEye",
                "mrz_failure_reason": "SUCCESS",
            }
        else:
            # MRZ not readable — either None or score too low.
            # Treat both as a quality failure: surface directly to user
            # instead of falling back to GPT hallucination.
            reason = "MRZ_LOW_SCORE" if (mrz_result is not None and _pe_score < 80) else "MRZ_NOT_READABLE"
            mrz_data["mrz_failure_reason"] = reason
            mrz_data["_quality_note"] = (
                "Passport MRZ could not be read — image quality is insufficient. "
                "Re-photograph the bio-data page flat on a surface, from directly above, "
                "in good lighting. Ensure the two lines of text at the bottom of the page "
                "are fully visible, horizontal, and in focus."
            )
    except Exception as e:
        mrz_data["_pe_error"]           = f"PassportEye error: {e}"
        mrz_data["mrz_failure_reason"]  = "UNAVAILABLE"

    # ── Step 3: GPT-4o-mini VIZ pass (always) + fallback for null MRZ fields ─
    mrz_nulls = [
        k for k in ("surname", "given_names", "passport_number",
                    "date_of_birth", "expiry_date", "nationality")
        if not mrz_data.get(k)
    ]
    needs_fallback = len(mrz_nulls) > 0

    _PASSPORT_SYSTEM = (
        "You are a precise document field extraction engine. You are given an image "
        "of a passport biodata page. Your job is to read specific fields from the "
        "printed visual data (VIZ) — the human-readable text printed above the "
        "machine-readable zone (MRZ) at the bottom of the page.\n\n"
        "Rules you must follow without exception:\n"
        "- Read characters exactly as printed. Do not correct, normalise, or infer.\n"
        "- All dates must be returned in YYYY-MM-DD format only.\n"
        "- If a field is not visible or not present, return null.\n"
        "- Never infer one field from another.\n"
        "- Return only valid JSON. No explanation, no markdown, no preamble."
    )

    fallback_note = (
        f"\nIMPORTANT: The MRZ parser failed — also extract these fields from "
        f"the printed VIZ area: {', '.join(mrz_nulls)}.\n"
        if needs_fallback else ""
    )

    _PASSPORT_USER = f"""{fallback_note}Extract the following fields from this passport image.

{{
  "surname": "Holder's surname as printed under 'Surname' label. Indian passports may have compound surnames (e.g. 'MANOJ RAJU'). In MRZ: characters between country code and << double chevron. Return uppercase or null.",
  "given_names": "Holder's given name(s) under 'Given Name(s)' label. In MRZ: characters after << double chevron, single < = space. Return uppercase or null.",
  "passport_no": "Passport number under 'Passport No.' label. Typically 1 letter + 7 digits. Do not confuse with file number or old passport number. Return uppercase or null.",
  "date_of_birth": "Date of birth under 'Date of Birth' label. Convert DD/MM/YYYY to YYYY-MM-DD. Return YYYY-MM-DD or null.",
  "expiry_date": "Expiry date under 'Date of Expiry' label. Convert DD/MM/YYYY to YYYY-MM-DD. Return YYYY-MM-DD or null.",
  "nationality": "Nationality under 'Nationality' label. Return 3-letter ISO code (IND, USA, GBR) or null.",
  "issue_date": "Date of issue under 'Date of Issue' label in the VIZ printed area. Always paired with 'Place of Issue'. Convert DD/MM/YYYY to YYYY-MM-DD. Read characters directly — do NOT calculate from expiry. '7' and '8' look similar in small print — read carefully. Return YYYY-MM-DD or null.",
  "place_of_issue": "City under 'Place of Issue' label. Do NOT confuse with 'Place of Birth' — these appear close together. Typically a major Indian city. Return uppercase or null.",
  "sex": "Sex under 'Sex' label. Return 'M', 'F', or null.",
  "mrz_line1": "Full first MRZ line exactly as printed including < characters, or null.",
  "mrz_line2": "Full second MRZ line exactly as printed including < characters, or null.",
  "surname_fallback": "Same as surname or null.",
  "given_names_fallback": "Same as given_names or null.",
  "passport_number_fallback": "Same as passport_no or null.",
  "date_of_birth_fallback": "Same as date_of_birth or null.",
  "expiry_date_fallback": "Same as expiry_date or null.",
  "nationality_fallback": "Same as nationality or null."
}}

Return only the JSON object. No explanation, no markdown."""

    viz_data = _gpt_extract(corrected_images, _PASSPORT_USER,  # all pages — bio-data may not be page 1
                            system_prompt=_PASSPORT_SYSTEM)

    # Write VIZ fallback values into mrz_data for any null MRZ fields
    if needs_fallback and isinstance(viz_data, dict):
        fallback_map = {
            "surname":         "surname_fallback",
            "given_names":     "given_names_fallback",
            "passport_number": "passport_number_fallback",
            "date_of_birth":   "date_of_birth_fallback",
            "expiry_date":     "expiry_date_fallback",
            "nationality":     "nationality_fallback",
        }
        for mrz_key, viz_key in fallback_map.items():
            fallback_val = viz_data.get(viz_key)
            if not mrz_data.get(mrz_key) and fallback_val and fallback_val != "null":
                mrz_data[mrz_key]             = fallback_val
                mrz_data[f"{mrz_key}_source"] = "GPT-VIZ-fallback"

    # ── Issue date: resolve 7/8 day-digit ambiguity (R-PASS-01) ──────────────
    # Indian passports expire on day BEFORE the 10-year anniversary of issue.
    # So: issue_date = expiry_date + 1 day, then subtract 10 years.
    # This is deterministic Python — not a prompt suggestion.
    issue_date  = viz_data.get("issue_date")  if isinstance(viz_data, dict) else None
    expiry_date = (mrz_data.get("expiry_date") or
                   (viz_data.get("expiry_date_fallback") if isinstance(viz_data, dict) else None))

    if issue_date and expiry_date and isinstance(issue_date, str) and len(issue_date) == 10:
        day = issue_date[8:10]
        if day in ("07", "08"):
            try:
                exp       = date.fromisoformat(expiry_date)
                issue_day = exp + timedelta(days=1)       # FIX: +1 day before -10 years
                try:
                    resolved = issue_day.replace(year=issue_day.year - 10)
                except ValueError:
                    resolved = issue_day.replace(year=issue_day.year - 10, day=28)
                viz_data["issue_date"]             = resolved.isoformat()
                viz_data["issue_date_review_flag"] = True
                viz_data["issue_date_review_reason"] = (
                    f"Day digit 7/8 ambiguity resolved via 10-year rule: "
                    f"expiry {expiry_date} + 1 day → issue {resolved.isoformat()}"
                )
            except Exception:
                viz_data["issue_date_review_flag"]   = True
                viz_data["issue_date_review_reason"] = (
                    "Day digit 7/8 ambiguity — verify against physical document"
                )

    mrz_data["mrz_verified"] = bool(mrz_data.get("checksum_valid"))
    return {"source": "passport", "mrz": mrz_data, "viz": viz_data}


def _extract_pan(file_bytes: bytes, filetype: str) -> dict:
    """
    PAN card extraction with 4th-character entity-type backstop (R-PAN-01).
    The 4th character of the PAN number is definitive: P = individual,
    other letters = company/entity. This overrides GPT visual classification.
    """
    images = _to_images(file_bytes, filetype)

    _PAN_SYSTEM = (
        "You are a precise document field extraction engine. You are given an image "
        "of an Indian PAN (Permanent Account Number) card issued by the Income Tax "
        "Department, Government of India. Your job is to read specific fields from "
        "the printed card.\n\n"
        "Rules you must follow without exception:\n"
        "- Read characters exactly as printed. Do not correct, normalise, or "
        "standardise spelling under any circumstances.\n"
        "- All dates must be returned in YYYY-MM-DD format only.\n"
        "- If a field is not visible, not present, or not applicable, return null.\n"
        "- Return only valid JSON. No explanation, no markdown, no preamble."
    )

    _PAN_USER = """Extract the following fields from this PAN card and return as JSON.

{
  "pan_name": "Full name exactly as printed under the Name label. Read every character verbatim — do NOT correct or normalise spelling. South Indian names ending in MURTHI, SWAMY, NAIDU, PILLAI etc. have legally distinct spellings — preserve them exactly. MURTHI (ending I) and MURTHY (ending Y) are different. Return uppercase string.",
  "pan_number": "10-character alphanumeric PAN number (5 letters, 4 digits, 1 letter — e.g. AWAPJ9107H). Return uppercase string.",
  "father_name": "Name under Father's Name / 'पिता का नाम' label. Apply same exact-read rule. Absent on company PANs — return null. Return uppercase or null.",
  "pan_dob": "Date of birth or incorporation under Date of Birth label. Convert DD/MM/YYYY to YYYY-MM-DD. Return YYYY-MM-DD or null.",
  "entity_type": "Return 'individual' if card shows a person photograph AND father's name field. Return 'company' if no photo and no father's name. Return 'individual' if photo present even if father name label unclear."
}

Return only the JSON object."""

    data = _gpt_extract(images, _PAN_USER, system_prompt=_PAN_SYSTEM)

    # R-PAN-01 hard backstop: 4th character of PAN number is definitive.
    # Overrides any GPT entity_type classification that contradicts it.
    # Older PAN card formats (pre-2017) omit English field labels — GPT may
    # misclassify them as company because it can't find "Father's Name" label.
    if isinstance(data, dict):
        pan_num = re.sub(r"[^A-Z0-9]", "", (data.get("pan_number") or "").upper())
        if len(pan_num) >= 4:
            fourth = pan_num[3]
            if fourth == "P" and data.get("entity_type") != "individual":
                data["entity_type"]          = "individual"
                data["entity_type_override"] = "4th_char_P_hard_rule"
            elif fourth in "FCHABLGJT" and data.get("entity_type") == "individual":
                data["entity_type"]          = "company"
                data["entity_type_override"] = "4th_char_non_P_hard_rule"

    return {"source": "pan", **(data if isinstance(data, dict) else {"error": "extraction failed"})}


def _extract_visa(file_bytes: bytes, filetype: str) -> dict:
    """
    Visa extraction with Tesseract MRZ read for expiry date (R-VISA-01).

    Sticker visa holographic overlays obscure VIZ date fields — the printed
    expiry date is unreliable. The MRZ lines on sticker visas are machine-printed
    outside the holographic zone and are reliably readable by Tesseract.

    MRZ line 2, positions 22-27 (1-indexed) = expiry date in YYMMDD format.
    This overrides any GPT VIZ date reading (R-VISA-01).
    """
    images = _to_images(file_bytes, filetype)
    corrected_images = [_correct_rotation(img) for img in images]

    mrz_data = {}

    # Step 1: Tesseract MRZ read — reliable for visa sticker MRZ.
    # FastMRZ targets passport MRZ (P< prefix); visa MRZ (VC/VA/VR) may not parse.
    # Binarize image first to improve OCR on degraded/holographic scans.
    try:
        from PIL import Image as PILImage
        gray = cv2.cvtColor(corrected_images[0], cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pil_img = PILImage.fromarray(binary)
        raw_text = pytesseract.image_to_string(
            pil_img,
            config="--psm 11 -c tessedit_char_whitelist="
                   "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
        )
        # Filter to lines that look like MRZ: >= 25 chars, contains < fill chars
        mrz_lines = [
            ln.strip().replace(" ", "")
            for ln in raw_text.splitlines()
            if len(ln.strip().replace(" ", "")) >= 25 and "<" in ln
        ]
        if len(mrz_lines) >= 2:
            line2 = mrz_lines[-1]
            if len(line2) >= 27:
                expiry_raw = line2[21:27]   # positions 22-27 (1-indexed) = 21-26 (0-indexed)
                if re.match(r"^[0-9]{6}$", expiry_raw):
                    yy, mm, dd = expiry_raw[:2], expiry_raw[2:4], expiry_raw[4:6]
                    century = "20" if int(yy) < 70 else "19"
                    mrz_data["expiry_date"]  = f"{century}{yy}-{mm}-{dd}"
                    mrz_data["_mrz_source"] = "Tesseract-MRZ"
            visa_raw = re.sub(r"[^A-Z0-9]", "", line2[:9])
            if len(visa_raw) >= 6:
                mrz_data["visa_number"] = visa_raw
    except Exception:
        pass   # Fall through — GPT-VIZ covers all fields as fallback

    # Step 2: GPT-4o-mini for VIZ fields not in MRZ
    _VISA_PROMPT = """Extract from this visa document.

STICKER VISA EXPIRY (Schengen, UK, US, and other sticker visas):
  Holographic overlays obscure VIZ date fields — the printed expiry date is unreliable.
  If MRZ is present (two OCR-B lines with < fill chars at bottom of sticker),
  read expiry from MRZ positions 22-27 (YYMMDD). E.g. '221205' = 2022-12-05.
  MRZ expiry ALWAYS overrides VIZ date.

DURATION-OF-STAY (e.g. '30', '90' on the sticker):
  This is a COUNT OF DAYS — NOT a date. Never use it as an expiry day, month, or year.
  Return it only in duration_days.

DESTINATION vs ISSUING COUNTRY:
  The country printed prominently on the sticker header is the ISSUER, not the destination.
  For Schengen visas (territory = 'Etats Schengen'): return 'Schengen' as destination.
  Only return a specific country if stated in visa purpose or remarks section.

ISSUING AUTHORITY:
  Look for city name printed on sticker body (e.g. 'New Delhi', 'Mumbai').
  Return as '[Country] Embassy/Consulate, [City]'. Or null.

Return ONLY this JSON:
{
  "visa_number": "visa number from sticker top or null",
  "destination_country": "'Schengen' for Schengen visas, or specific destination country if stated — NOT the issuing country from sticker header",
  "issue_date": "valid-from date in YYYY-MM-DD or null",
  "expiry_date": "MRZ positions 22-27 preferred (YYMMDD to YYYY-MM-DD), else VIZ valid-until — YYYY-MM-DD or null",
  "visa_type": "TOURIST / BUSINESS / STUDENT / TRANSIT / MULTIPLE or null",
  "issuing_authority": "[Country] Embassy/Consulate, [City] or null",
  "duration_days": "days-of-stay count as string e.g. '30' — NOT a date — or null",
  "entries": "MULT or SINGLE or null"
}"""

    data = _gpt_extract(corrected_images, _VISA_PROMPT)
    if not isinstance(data, dict):
        data = {}

    # Step 3: Tesseract MRZ wins for date fields (R-VISA-01 hard rule)
    if mrz_data.get("expiry_date"):
        data["expiry_date"] = mrz_data["expiry_date"]
    if mrz_data.get("visa_number") and not data.get("visa_number"):
        data["visa_number"] = mrz_data["visa_number"]
    if mrz_data.get("_mrz_source"):
        data["_mrz_source"] = mrz_data["_mrz_source"]

    # Clean up issuing_authority — GPT may return "[Country], null" when city absent
    if isinstance(data, dict) and data.get("issuing_authority"):
        auth = str(data["issuing_authority"]).strip()
        # Remove trailing ", null" or ", None" artifacts
        auth = re.sub(r",\s*(null|None)\s*$", "", auth, flags=re.IGNORECASE).strip()
        data["issuing_authority"] = auth if auth.lower() not in ("null", "none", "") else None

    return {"source": "visa", **data}


# ── Lookup tables for ticket extraction ───────────────────────────────────────

_INDIAN_AIRPORTS = {
    "DEL", "BOM", "MAA", "HYD", "BLR", "CCU", "AMD", "COK", "GOI",
    "PNQ", "JAI", "ATQ", "IXC", "TRV", "IXM", "CJB", "VTZ", "SXR",
    "GAU", "BBI", "IXR", "PAT", "NAG", "IDR", "BHO", "VNS", "LKO",
    "IXB", "IXS", "IXU", "JDH", "JSA", "UDR", "JLR", "RPR", "BDQ",
}

# IATA code → country name (for destination_country derivation)
_IATA_COUNTRY = {
    "PAR": "France",  "CDG": "France",  "ORY": "France",
    "LHR": "United Kingdom", "LGW": "United Kingdom", "LCY": "United Kingdom",
    "MXP": "Italy",   "FCO": "Italy",   "NAP": "Italy",  "MIL": "Italy",
    "FRA": "Germany", "MUC": "Germany", "BER": "Germany", "TXL": "Germany",
    "AMS": "Netherlands",
    "MAD": "Spain",   "BCN": "Spain",
    "ZRH": "Switzerland", "GVA": "Switzerland",
    "VIE": "Austria",
    "BRU": "Belgium",
    "CPH": "Denmark",
    "ARN": "Sweden",  "STO": "Sweden",
    "HEL": "Finland",
    "OSL": "Norway",
    "ATH": "Greece",
    "IST": "Turkey",  "SAW": "Turkey",
    "PRG": "Czech Republic",
    "WAW": "Poland",
    "BUD": "Hungary",
    "LIS": "Portugal",
    "EDI": "United Kingdom", "MAN": "United Kingdom",
    "DXB": "UAE",     "AUH": "UAE",    "SHJ": "UAE",
    "DOH": "Qatar",
    "RUH": "Saudi Arabia", "JED": "Saudi Arabia",
    "MCT": "Oman",
    "BAH": "Bahrain",
    "KWI": "Kuwait",
    "SIN": "Singapore",
    "KUL": "Malaysia",
    "BKK": "Thailand", "HKT": "Thailand",
    "HKG": "Hong Kong",
    "NRT": "Japan",   "KIX": "Japan",   "TYO": "Japan",
    "ICN": "South Korea",
    "PEK": "China",   "PVG": "China",
    "SYD": "Australia", "MEL": "Australia",
    "JFK": "USA",     "EWR": "USA",    "LAX": "USA",
    "ORD": "USA",     "SFO": "USA",    "MIA": "USA",
    "YYZ": "Canada",  "YVR": "Canada",
    "NBO": "Kenya",
    "JNB": "South Africa",
    "CAI": "Egypt",
    "CMN": "Morocco",
}

# City name (as printed in itineraries) → IATA code
# GPT reliably reads city names from print; IATA conversion is deterministic.
_CITY_TO_IATA = {
    "DELHI": "DEL",    "NEW DELHI": "DEL",
    "MUMBAI": "BOM",   "BOMBAY": "BOM",
    "CHENNAI": "MAA",  "MADRAS": "MAA",
    "HYDERABAD": "HYD",
    "BANGALORE": "BLR", "BENGALURU": "BLR",
    "KOLKATA": "CCU",  "CALCUTTA": "CCU",
    "AHMEDABAD": "AMD",
    "KOCHI": "COK",    "COCHIN": "COK",
    "GOA": "GOI",
    "PUNE": "PNQ",
    "JAIPUR": "JAI",
    "AMRITSAR": "ATQ",
    "TRIVANDRUM": "TRV", "THIRUVANANTHAPURAM": "TRV",
    "MADURAI": "IXM",
    "COIMBATORE": "CJB",
    "VISAKHAPATNAM": "VTZ",
    "GUWAHATI": "GAU",
    "BHUBANESWAR": "BBI",
    "LUCKNOW": "LKO",
    "PATNA": "PAT",
    "NAGPUR": "NAG",
    "INDORE": "IDR",
    "BHOPAL": "BHO",
    "VARANASI": "VNS",
    "DUBAI": "DXB",    "ABU DHABI": "AUH", "SHARJAH": "SHJ",
    "DOHA": "DOH",
    "RIYADH": "RUH",   "JEDDAH": "JED",
    "MUSCAT": "MCT",
    "KUWAIT": "KWI",
    "BAHRAIN": "BAH",
    "PARIS": "CDG",
    "LONDON": "LHR",
    "FRANKFURT": "FRA",
    "AMSTERDAM": "AMS",
    "MILAN": "MXP",
    "ROME": "FCO",
    "MADRID": "MAD",   "BARCELONA": "BCN",
    "ZURICH": "ZRH",   "GENEVA": "GVA",
    "VIENNA": "VIE",
    "BRUSSELS": "BRU",
    "COPENHAGEN": "CPH",
    "STOCKHOLM": "ARN",
    "OSLO": "OSL",
    "HELSINKI": "HEL",
    "ATHENS": "ATH",
    "ISTANBUL": "IST",
    "PRAGUE": "PRG",
    "WARSAW": "WAW",
    "BUDAPEST": "BUD",
    "LISBON": "LIS",
    "EDINBURGH": "EDI", "MANCHESTER": "MAN",
    "SINGAPORE": "SIN",
    "KUALA LUMPUR": "KUL",
    "BANGKOK": "BKK",  "PHUKET": "HKT",
    "HONG KONG": "HKG",
    "TOKYO": "NRT",    "OSAKA": "KIX",
    "SEOUL": "ICN",
    "BEIJING": "PEK",  "SHANGHAI": "PVG",
    "SYDNEY": "SYD",   "MELBOURNE": "MEL",
    "NEW YORK": "JFK", "LOS ANGELES": "LAX",
    "CHICAGO": "ORD",  "SAN FRANCISCO": "SFO", "MIAMI": "MIA",
    "TORONTO": "YYZ",  "VANCOUVER": "YVR",
    "NAIROBI": "NBO",
    "JOHANNESBURG": "JNB",
    "CAIRO": "CAI",
    "CASABLANCA": "CMN",
}


def _city_to_iata(city: str) -> str:
    """Convert a printed city name to IATA code. Falls back to city[:3] if unknown."""
    city = city.upper().strip()
    # Direct lookup
    code = _CITY_TO_IATA.get(city, "")
    if not code:
        # Partial match — handles 'DELHI INDIA', 'NEW DELHI INDIA' etc.
        for known, known_code in _CITY_TO_IATA.items():
            if known in city or city in known:
                code = known_code
                break
    if not code and len(city) == 3 and city.isalpha():
        code = city   # Already looks like an IATA code
    return code or city[:3].upper()


def _extract_ticket(
    file_bytes: bytes,
    filetype: str,
    applicant_name: str = None,
) -> dict:
    """
    Ticket extraction with structured leg parsing (R-TICKET-06) and
    all_passengers list (R-TICKET-07).

    applicant_name: Optional. If passport has already been uploaded and
    extracted, pass 'SURNAME/GIVENNAMES' here so GPT can locate the
    specific passenger row in a group booking table.

    Step 1a: LLM extracts legs using CITY NAMES (more reliable than IATA
             codes from ambiguous itinerary email text). Python converts to IATA.
    Step 1b: LLM extracts all_passengers from group booking table.
    Step 2:  Python derives departure_date_raw, return_date_raw, origin_iata,
             destination_iata, destination_country deterministically.
    """
    images = _to_images(file_bytes, filetype)

    # ── Step 1a: Extract legs using city names ────────────────────────────────
    legs_prompt = """List every flight leg in this itinerary in order.
Return ONLY a JSON array — no other text:
[
  {"leg": 1, "flight": "EK515", "from_city": "DELHI",  "to_city": "DUBAI", "date_raw": "09NOV", "dep_time": "21:25"},
  {"leg": 2, "flight": "EK071", "from_city": "DUBAI",  "to_city": "PARIS", "date_raw": "10NOV", "dep_time": "03:20"},
  {"leg": 3, "flight": "EK092", "from_city": "MILAN",  "to_city": "DUBAI", "date_raw": "21NOV", "dep_time": "21:35"},
  {"leg": 4, "flight": "EK516", "from_city": "DUBAI",  "to_city": "DELHI", "date_raw": "22NOV", "dep_time": "09:55"}
]
Use the CITY NAME exactly as printed for from_city and to_city — do NOT convert to IATA codes.
date_raw is the departure date exactly as printed (e.g. "09NOV", "10-NOV", "09/11/2022").
Return [] if no flights found."""

    legs_raw = _gpt_extract(images, legs_prompt)
    raw_legs = legs_raw if isinstance(legs_raw, list) else []

    # Convert city names to IATA codes (R-TICKET-06 Fix B)
    legs = []
    for raw_leg in raw_legs:
        leg = dict(raw_leg)
        for city_key, iata_key in [("from_city", "from_iata"), ("to_city", "to_iata")]:
            city = str(leg.pop(city_key, "") or leg.get(iata_key, "") or "").upper().strip()
            leg[iata_key] = _city_to_iata(city)
        legs.append(leg)

    # ── Step 1b: Extract all passengers (R-TICKET-07) ────────────────────────
    _appl_hint = (
        f"\n\nAPPLICANT: The travel card is being issued to '{applicant_name}'. "
        "Find this person in the passenger table (Col 7) and record their row "
        "in the applicant_row field."
        if applicant_name else ""
    )

    pax_prompt = f"""Extract ALL passenger rows from this flight itinerary or booking confirmation.

GROUP BOOKING EMAIL — TABLE COLUMN STRUCTURE:
  Col 1: Group booking reference (long alphanumeric, e.g. ZVE245101122) — NOT a PNR
  Col 2: Route type (e.g. Ex-India)
  Col 3: Departure airport IATA (e.g. DEL)
  Col 4: Travel agent / booking agent name — AGENT, NOT the passenger
  Col 5: Individual PNR (6-10 alphanumeric, e.g. CGT2206204)
  Col 6: Passenger type (Adult / Child)
  Col 7: Passenger name as per passport — SURNAME/GIVENNAME format (e.g. JAIN/SHUBHAM Mr)

CRITICAL:
  Col 7 at the END of each row = ACTUAL PASSENGER.
  Col 4 in the MIDDLE = TRAVEL AGENT — do NOT list as passenger.
  Email sender name and signature = travel agent — do NOT list as passenger.
  Extract EVERY row — do not stop after first few.{_appl_hint}

Return ONLY this JSON:
{{
  "all_passengers": [
    {{"name": "JAIN/SHUBHAM", "pnr": "CGT2206204"}},
    {{"name": "KUMAR/RAJEEV", "pnr": "CGS2204024"}}
  ],
  "applicant_row": {{"name": "JAIN/SHUBHAM", "pnr": "CGT2206204"}} or null,
  "airline": "airline name or null",
  "doc_issue_date": "YYYY-MM-DD or null"
}}"""

    pax_raw = _gpt_extract(images, pax_prompt, max_tokens=1000)
    pax = pax_raw if isinstance(pax_raw, dict) else {}
    all_passengers = pax.get("all_passengers") or []
    if not isinstance(all_passengers, list):
        all_passengers = []

    # Use applicant_row if model found the applicant directly
    applicant_row = pax.get("applicant_row")
    if isinstance(applicant_row, dict) and applicant_row.get("name"):
        _initial_name = applicant_row["name"]
        _initial_pnr  = applicant_row.get("pnr")
    elif all_passengers:
        _initial_name = all_passengers[0].get("name")
        _initial_pnr  = all_passengers[0].get("pnr")
    else:
        _initial_name = None
        _initial_pnr  = None

    # PNR validation: IATA PNR = 5-8 alphanumeric characters.
    # Anything outside this range is a phone number, flight number,
    # or other field grabbed by mistake on degraded scans — null it out.
    def _valid_pnr(v):
        if not v:
            return None
        clean = re.sub(r"[^A-Z0-9]", "", str(v).upper())
        return clean if 5 <= len(clean) <= 10 else None

    _initial_pnr = _valid_pnr(_initial_pnr)
    for _pax_row in all_passengers:
        _pax_row["pnr"] = _valid_pnr(_pax_row.get("pnr"))

    # ── Step 2: Programmatic derivation (R-TICKET-06) ────────────────────────
    departure_date_raw  = None
    return_date_raw     = None
    origin_iata         = None
    destination_iata    = None
    destination_country = None

    if legs:
        # First departure from India
        for leg in legs:
            if str(leg.get("from_iata", "")).upper() in _INDIAN_AIRPORTS:
                origin_iata        = leg["from_iata"].upper()
                departure_date_raw = leg.get("date_raw")
                break

        # Last arrival into India (return leg)
        for leg in reversed(legs):
            if str(leg.get("to_iata", "")).upper() in _INDIAN_AIRPORTS:
                return_date_raw = leg.get("date_raw")
                break

        # Destination: furthest outbound city before the journey turns back.
        #
        # Algorithm — visited-set approach:
        #   Walk legs in order. Track every city seen so far.
        #   The outbound journey continues as long as each new arrival city
        #   has NOT been visited before (i.e. it is a genuinely new stop).
        #   The moment a leg arrives at a city already in the visited set,
        #   the journey has turned around — the previous to_iata is destination.
        #
        # Examples:
        #   DEL->IST->BCN->IST->DEL : visits {DEL,IST,BCN}, IST revisited at Leg3 -> BCN
        #   DEL->DXB->PAR->DEL      : visits {DEL,DXB,PAR}, DEL revisited at Leg3 -> PAR
        #   DEL->DXB->BCN->MAD->DXB->DEL : visits {DEL,DXB,BCN,MAD}, DXB revisited -> MAD
        #   DEL->LHR->JFK->LHR->DEL : visits {DEL,LHR,JFK}, LHR revisited at Leg3 -> JFK
        visited = set()
        last_outbound_to = None
        if legs:
            visited.add(str(legs[0].get("from_iata", "")).upper())
        for leg in legs:
            to = str(leg.get("to_iata", "")).upper()
            if not to:
                continue
            if to in visited:
                # Journey has turned back — last_outbound_to is the destination
                break
            visited.add(to)
            if to not in _INDIAN_AIRPORTS:
                last_outbound_to = to

        if last_outbound_to:
            destination_iata    = last_outbound_to
            destination_country = _IATA_COUNTRY.get(destination_iata)
        else:
            # Fallback: first non-Indian to_iata (handles one-way tickets)
            for leg in legs:
                to = str(leg.get("to_iata", "")).upper()
                if to not in _INDIAN_AIRPORTS and to:
                    destination_iata    = to
                    destination_country = _IATA_COUNTRY.get(destination_iata)
                    break

    return {
        "source":               "ticket",
        "legs":                 legs,
        "all_passengers":       all_passengers,
        # all_date_strings: raw dates from legs, used for visa coverage check
        # and year inference in cross-validation — no year guessing needed.
        "all_date_strings":     sorted({
            leg.get("date_raw", "").strip()
            for leg in legs if leg.get("date_raw")
        }),
        # passenger_name and pnr: initially set to applicant_row or first passenger.
        # Cross-validation updates these via passport-name fuzzy match (R-TICKET-07).
        "passenger_name":       _initial_name,
        "pnr":                  _initial_pnr,
        "airline":              pax.get("airline"),
        "doc_issue_date":       pax.get("doc_issue_date"),
        "origin_iata":          origin_iata,
        "destination_iata":     destination_iata,
        "destination_country":  destination_country,
        "departure_date_raw":   departure_date_raw,
        "return_date_raw":      return_date_raw,
        # departure_date / return_date resolved in cross-validation using
        # visa window as year anchor (R-XVAL visa coverage check).
        "departure_date":       None,
        "return_date":          None,
    }


def _extract_reload(file_bytes: bytes, filetype: str) -> dict:
    images = _to_images(file_bytes, filetype)

    data   = _gpt_extract(images, """Extract from a Thomas Cook Prepaid Travel Card application/reload form.

FORM FORMAT — read carefully before extracting:
- Handwritten fields use individual letter boxes (one character per box) — read each box as one character and join them
- One page is the DECLARATION page — it has 'Cardholder Declaration' heading, dense legal text,
  and a signature/date block at the bottom. Extract: cardholder name (printed), place, date from here.
- One page is the DATA ENTRY page — it has labelled grid boxes for First Name, Last Name,
  Passport Number, Date of Birth, PAN Number, Address, Card Number, etc.
  Extract all identity and card fields from this page.
- Do not assume which page comes first — identify each page by its content.

CRITICAL — the form contains TWO separate tables. You must distinguish them:

  TABLE 1 — Customer currency request (what you extract from):
    Column headers: "Currency" | "Amount"
    The Amount column has handwritten values where the customer filled in
    how much of each currency they want (e.g. Euro (EUR) → 1000/=)
    Blank rows in Amount = customer did NOT request that currency.

  TABLE 2 — Bank fees table (ignore entirely):
    Column headers: "Currency" | "Balance Enquiry Fee" | "ATM Withdrawal Fees"
    This shows bank charges like USD 0.50, EUR 0.50 — these are NOT
    the customer's requested amounts. Do not read from this table at all.

You can identify which table you are reading by checking the column headers.
Only extract currencies from Table 1 where a handwritten amount exists.

Return ONLY this JSON (use null for any field not found):
{
  "cardholder_name": "full name as written",
  "first_name": "first name from form field",
  "last_name": "last name from form field",
  "passport_number": "passport number from form",
  "passport_issue_date": "YYYY-MM-DD",
  "passport_expiry": "YYYY-MM-DD",
  "place_of_issue": "city",
  "date_of_birth": "YYYY-MM-DD",
  "pan_number": "PAN number from form",
  "address": "full address as written",
  "pin_code": "PIN code",
  "mobile": "mobile number",
  "email": "email address",
  "card_number": "16-digit card number or null",
  "purpose_of_travel": "as written e.g. Tourism",
  "currencies_requested": ["list ONLY currencies with a handwritten non-zero amount, e.g. [\"EUR\"]"],
  "currency_amounts": {"EUR": "1000", "USD": "500"},
  "card_type": "Borderless Prepaid Card or One Currency Card or null",
  "card_network": "Mastercard or Visa or null",
  "date": "YYYY-MM-DD from signature date field",
  "place": "place from signature section"
}"""
    )
    # Override currencies_requested from currency_amounts when available.
    # currency_amounts (dict) is more reliable than currencies_requested (list)
    # because the prompt extracts amounts next to each currency row — the fees
    # table on the right cannot produce a non-null amount there.
    amt_map = data.get("currency_amounts", {})
    if amt_map:
        derived = [
            k.upper() for k, v in amt_map.items()
            if v and str(v).strip() not in ("", "null", "0", "None", "0.0")
        ]
        if derived:
            data["currencies_requested"] = derived
    return {"source": "reload", **data}


def _extract_transfer(file_bytes: bytes, filetype: str) -> dict:
    images = _to_images(file_bytes, filetype)
    data   = _gpt_extract(images, """Extract from the bank transfer acknowledgement.
Return ONLY this JSON:
{
  "amount_inr": "numeric amount in INR as string or null",
  "reference_number": "UTR or reference number or null",
  "transfer_date": "YYYY-MM-DD or null",
  "beneficiary": "recipient name or null"
}""")
    return {"source": "transfer", **data}


def _extract_annexure(file_bytes: bytes, filetype: str) -> dict:
    images = _to_images(file_bytes, filetype)
    data   = _gpt_extract(images, """Extract from the annexure/dealer declaration letter.
Return ONLY this JSON:
{
  "dealer_name": "forex dealer company name or null",
  "rbi_licence": "RBI licence number e.g. DEL.FFMC/861/2020 or null",
  "cardholder_name": "card holder name in letter or null",
  "currency": "currency code or null",
  "amount": "amount or null",
  "utr_reference": "UTR or reference number or null"
}""")
    return {"source": "annexure", **data}


def _extract_dealer_issuance_letter(file_bytes: bytes, filetype: str) -> dict:
    """
    R-ANNEXURE-01: Forex Issuance Letter extractor.

    A dealer issuance letter (from forex sub-dealer to Thomas Cook / authorised
    dealer) contains a structured table with 10+ cross-validatable fields.
    travel_date is the most reliable year anchor for undated ticket dates.
    The UTR number substitutes for a standalone bank transfer document.
    """
    images = _to_images(file_bytes, filetype)
    data   = _gpt_extract(images, """Extract from this forex dealer issuance letter.
This letter is from a sub-dealer addressed to Thomas Cook India Ltd.
It contains a structured table with card holder details.

Return ONLY this JSON:
{
  "cardholder_name": "card holder name from table or null",
  "card_number": "forex card number from table or null",
  "passport_no": "passport number from table or null",
  "pan_number": "PAN number from table (may be labeled BTQ PAN) or null",
  "dob": "date of birth in YYYY-MM-DD from table or null",
  "doi": "date of issue of passport in YYYY-MM-DD from table or null",
  "doe": "date of expiry of passport in YYYY-MM-DD from table or null",
  "mother_name": "mother name from table or null",
  "travel_date": "travel date EXACTLY as printed — AUTHORITATIVE year anchor for ticket dates. Return as string e.g. '09-11-2022'. Do NOT convert format. Or null",
  "destination": "destination from table or null",
  "currency_code": "currency code (USD/EUR/GBP etc.) or null",
  "amount_foreign": "foreign currency amount or null",
  "amount_inr": "INR equivalent amount or null",
  "utr_number": "UTR or payment reference number or null",
  "activation_or_reload": "ACTIVATION or RELOAD or null",
  "travel_type": "BTQ or BUSINESS or null",
  "dealer_name": "issuing dealer company name from letterhead or null",
  "letter_date": "date of this letter in YYYY-MM-DD or null"
}""")
    return {"source": "dealer_issuance_letter", **data}


def _extract_sof(file_bytes: bytes, filetype: str) -> dict:
    """
    SOF extraction. Handles both bank account statements (preferred)
    and single-transaction bank advice receipts (R-SOF-01).
    """
    images = _to_images(file_bytes, filetype)
    data   = _gpt_extract(images, """Extract from this bank document.
This may be a bank account statement (multi-transaction, preferred for SOF)
or a single-transaction bank advice/receipt.

Return ONLY this JSON:
{
  "account_holder_name": "name as printed on account or null",
  "bank_name": "bank name or null",
  "account_number": "account number or null",
  "statement_from": "statement period start YYYY-MM-DD or null",
  "statement_to": "statement period end YYYY-MM-DD or null",
  "closing_balance": "closing balance as string with currency or null",
  "document_variant": "'bank_statement' if multi-transaction statement, 'transfer_receipt' if single-transaction advice/receipt",
  "transaction_date": "for transfer_receipt: transaction date YYYY-MM-DD or null",
  "transaction_amount": "for transfer_receipt: amount as string or null",
  "utr_reference": "for transfer_receipt: UTR or transaction ID or null"
}""")
    return {"source": "sof", **data}


# ── Extraction dispatcher ──────────────────────────────────────────────────────

_EXTRACTORS = {
    "passport":               _extract_passport,
    "pan":                    _extract_pan,
    "visa":                   _extract_visa,
    "ticket":                 _extract_ticket,
    "reload":                 _extract_reload,
    "transfer":               _extract_transfer,
    "annexure":               _extract_annexure,
    "sof":                    _extract_sof,
    "dealer_issuance_letter": _extract_dealer_issuance_letter,
}


# ── Currency-destination matrix ────────────────────────────────────────────────

_UNIVERSAL = {"USD", "EUR", "GBP"}   # always pass regardless of destination

_STRICT = {
    "JPY": ["Japan"],
    "THB": ["Thailand"],
    "SGD": ["Singapore"],
    "CAD": ["Canada"],
    "CHF": ["Switzerland"],
    "AUD": ["Australia", "New Zealand"],
    "AED": ["United Arab Emirates", "UAE", "Saudi Arabia", "Qatar",
            "Bahrain", "Kuwait", "Oman"],
}


# ── Cross-validation helpers ───────────────────────────────────────────────────

def _resolve_date_year(date_str: str, visa_from: date, visa_until: date):
    """
    Infer the calendar year for a printed date string (e.g. '09NOV') using the
    visa validity window as the anchor. No LLM involvement — fully deterministic.

    Strategy: try candidate years spanning the visa window ±1. Pick the year
    that places the date closest to the centre of the visa window.

    Examples:
      '09NOV' + visa 2022-11-06 to 2022-12-05  →  2022-11-09  ✅
      '22NOV' + visa 2022-11-06 to 2022-12-05  →  2022-11-22  ✅
      '03JAN' + visa 2022-12-15 to 2023-01-15  →  2023-01-03  ✅
    """
    months = {m.upper(): i for i, m in enumerate(month_abbr) if m}
    m = re.match(r"(\d{1,2})\s*([A-Za-z]{3})", str(date_str).strip())
    if not m:
        return None
    try:
        day = int(m.group(1))
        mon = months.get(m.group(2).upper())
        if not mon:
            return None
        yrs = {visa_from.year - 1, visa_from.year,
               visa_until.year, visa_until.year + 1}
        mid = visa_from + (visa_until - visa_from) // 2
        best, best_dist = None, float("inf")
        for yr in sorted(yrs):
            try:
                candidate = date(yr, mon, day)
                dist = abs((candidate - mid).days)
                if dist < best_dist:
                    best_dist, best = dist, candidate
            except ValueError:
                continue
        return best
    except Exception:
        return None


# ── Cross-validation ───────────────────────────────────────────────────────────

def _cross_validate(extracted: dict) -> list:
    checks = []
    today  = date.today()

    passport = extracted.get("passport", {})
    pan      = extracted.get("pan", {})
    visa     = extracted.get("visa", {})
    ticket   = extracted.get("ticket", {})
    reload   = extracted.get("reload", {})
    transfer = extracted.get("transfer", {})
    annexure = extracted.get("annexure", {})
    sof      = extracted.get("sof", {})

    # Collect dealer issuance letter — direct or from wrong-slot extraction
    issuance = extracted.get("dealer_issuance_letter", {})
    for k, v in extracted.items():
        if k.startswith("_supporting_") and isinstance(v, dict):
            if v.get("source") == "dealer_issuance_letter":
                issuance = v
                break

    mrz = passport.get("mrz", {})
    viz = passport.get("viz", {})

    # ── Identity source decision ──────────────────────────────────────────────
    # When passport MRZ is unverified (PassportEye could not validate it),
    # the passport name comes from GPT reading a borderline quality scan and
    # may be unreliable (e.g. hallucinated surnames from image noise).
    # PAN is government-issued, machine-printed, and consistently reliable.
    # When MRZ is unverified AND PAN is available, use PAN name as the
    # primary identity anchor for cross-validation comparisons.
    _mrz_verified    = bool(mrz.get("mrz_verified") or mrz.get("checksum_valid"))
    _pan_name        = pan.get("pan_name", "").upper().strip() if pan.get("pan_name") else None
    _use_pan_primary = not _mrz_verified and bool(_pan_name)

    # ── 0: Visa-window year inference + all-dates coverage check ──────────────
    # The visa valid_from/valid_until window is the most reliable year anchor
    # for undated ticket dates. No LLM year guessing needed.
    # Side-effect: sets ticket["departure_date"] and ["return_date"] with the
    # correct year — the passport validity check (below) uses these.
    _visa_from_str  = visa.get("issue_date")
    _visa_until_str = visa.get("expiry_date")
    _all_date_strs  = (
        ticket.get("all_date_strings")
        or [l.get("date_raw") for l in ticket.get("legs", []) if l.get("date_raw")]
    )

    if _all_date_strs and _visa_from_str and _visa_until_str:
        try:
            _vf = date.fromisoformat(_visa_from_str)
            _vu = date.fromisoformat(_visa_until_str)
            _resolved = sorted(filter(None, [
                _resolve_date_year(ds, _vf, _vu) for ds in _all_date_strs
            ]))
            if _resolved:
                ticket["departure_date"]             = _resolved[0].isoformat()
                ticket["return_date"]                = _resolved[-1].isoformat()
                ticket["departure_date_year_source"] = "visa_window"

                _outside = [d for d in _resolved if not (_vf <= d <= _vu)]
                if _outside:
                    checks.append({
                        "check":  "Visa Covers All Travel Dates",
                        "result": "warning",
                        "detail": (
                            f"Visa valid {_visa_from_str} to {_visa_until_str}. "
                            f"Date(s) outside window: {[str(d) for d in _outside]}. "
                            f"Verify — may be multi-entry or transit destination."
                        ),
                    })
                else:
                    checks.append({
                        "check":  "Visa Covers All Travel Dates",
                        "result": "pass",
                        "detail": (
                            f"Visa {_visa_from_str} to {_visa_until_str} covers all "
                            f"{len(_resolved)} travel date(s): "
                            f"{_resolved[0]} to {_resolved[-1]}."
                        ),
                    })
        except Exception as _ex:
            checks.append({
                "check":  "Visa Covers All Travel Dates",
                "result": "warning",
                "detail": f"Could not verify visa coverage: {_ex}",
            })
    elif _all_date_strs and not _visa_from_str:
        # No visa — try year anchors in priority order:
        # 1. Dealer issuance letter travel_date
        # 2. Ticket's own doc_issue_date (always on the ticket, no external doc needed)
        _MN = {"JAN":"01","FEB":"02","MAR":"03","APR":"04",
               "MAY":"05","JUN":"06","JUL":"07","AUG":"08",
               "SEP":"09","OCT":"10","NOV":"11","DEC":"12"}

        def _apply_year(yr_str: str, source_label: str):
            resolved_any = False
            for raw, field in [
                (ticket.get("departure_date_raw", ""), "departure_date"),
                (ticket.get("return_date_raw",    ""), "return_date"),
            ]:
                if not raw:
                    continue
                m2 = re.search(r"(\d{1,2})[^\d]?([A-Za-z]{3})", raw)
                if m2:
                    mon2 = _MN.get(m2.group(2).upper())
                    if mon2:
                        ticket[field] = f"{yr_str}-{mon2}-{m2.group(1).zfill(2)}"
                        resolved_any  = True
            if resolved_any:
                checks.append({
                    "check":  "Ticket Year Inference",
                    "result": "pass",
                    "detail": (
                        f"Travel year inferred from {source_label}: "
                        f"departure {ticket.get('departure_date')}, "
                        f"return {ticket.get('return_date')}."
                    ),
                })
            return resolved_any

        # Priority 1: dealer issuance letter
        travel_date_str = issuance.get("travel_date") if issuance else None
        _year_resolved  = False
        if travel_date_str and ticket.get("departure_date_raw"):
            try:
                parts = travel_date_str.replace("/", "-").split("-")
                if len(parts) == 3 and len(parts[2]) == 4:
                    _year_resolved = _apply_year(parts[2], "dealer issuance letter")
            except Exception:
                pass

        # Priority 2: ticket's own issue date (e.g. "03Nov22" or "2022-11-03")
        # Always present on the ticket — reliable year anchor when no other doc available.
        if not _year_resolved and ticket.get("doc_issue_date"):
            try:
                issue_iso = ticket["doc_issue_date"]
                if len(issue_iso) >= 4:
                    yr_str = issue_iso[:4]    # YYYY from YYYY-MM-DD
                    if yr_str.isdigit() and 2000 <= int(yr_str) <= 2040:
                        _year_resolved = _apply_year(yr_str, "ticket issue date")
            except Exception:
                pass

    # ── 1: Passport validity vs travel date (not today) ───────────────────────
    # Comparing against today causes false failures for historical applications
    # where the passport has since expired but was valid at time of travel.
    passport_expiry = mrz.get("expiry_date") or viz.get("expiry_date_fallback")
    if passport_expiry:
        try:
            exp = date.fromisoformat(passport_expiry)
            departure_date = ticket.get("departure_date")
            travel_date = None
            if departure_date:
                try:
                    travel_date = date.fromisoformat(departure_date)
                except Exception:
                    pass
            check_date = travel_date or today

            if exp < check_date:
                if travel_date:
                    days_expired = abs((travel_date - exp).days)
                    checks.append({
                        "check":  "Passport Validity at Travel",
                        "result": "fail",
                        "detail": (
                            f"Passport expired {days_expired} day(s) before departure "
                            f"(expiry: {passport_expiry}, departure: {travel_date}) — "
                            f"HARD FAIL: passport was not valid for this trip."
                        ),
                    })
                else:
                    days_expired = (today - exp).days
                    checks.append({
                        "check":  "Passport Validity",
                        "result": "warning",
                        "detail": (
                            f"Passport has expired ({passport_expiry}, {days_expired} days ago). "
                            f"No departure date available to confirm validity at time of travel — "
                            f"verify manually."
                        ),
                    })
            elif exp < today and travel_date and exp >= travel_date:
                # Expired since travel but was valid at departure — historical application
                checks.append({
                    "check":  "Passport Validity at Travel",
                    "result": "pass",
                    "detail": (
                        f"Passport was valid at departure ({travel_date}) — "
                        f"has since expired ({passport_expiry}). Historical application."
                    ),
                })
            else:
                days_remaining = (exp - today).days
                checks.append({
                    "check":  "Passport Validity",
                    "result": "pass",
                    "detail": f"Valid until {passport_expiry} ({days_remaining} days remaining).",
                })
        except Exception:
            checks.append({
                "check":  "Passport Validity",
                "result": "warning",
                "detail": f"Could not parse expiry date: {passport_expiry}",
            })

    # ── 2: Name match — passport vs PAN ──────────────────────────────────────
    passport_name_full    = None
    passport_name_surname = None

    if mrz.get("given_names") and mrz.get("surname"):
        passport_name_full    = (mrz["given_names"] + " " + mrz["surname"]).upper().strip()
        passport_name_surname = mrz["surname"].upper().strip()
    elif mrz.get("surname"):
        passport_name_full    = mrz["surname"].upper().strip()
        passport_name_surname = mrz["surname"].upper().strip()
    elif viz.get("full_name_viz"):
        passport_name_full = viz["full_name_viz"].upper().strip()

    pan_name = pan.get("pan_name", "").upper().strip() if pan.get("pan_name") else None

    # When MRZ unverified + PAN available: use PAN as the identity anchor.
    # Downstream comparisons (ticket, reload, SOF) will use this name.
    if _use_pan_primary:
        _passport_name_for_display = passport_name_full   # keep for showing what passport said
        passport_name_full    = _pan_name                 # PAN is more reliable
        passport_name_surname = _pan_name.split()[-1] if _pan_name else None
        checks.append({
            "check":  "Name Match — Passport vs PAN",
            "result": "warning",
            "detail": (
                f"Passport MRZ unverified — GPT read '{_passport_name_for_display}' "
                f"from borderline quality scan. Using PAN name '{_pan_name}' as "
                f"primary identity for this application. Verify passport name "
                f"against physical document."
            ),
        })

    elif passport_name_full and pan_name:
        try:
            sim_full    = rfdist.JaroWinkler.similarity(passport_name_full, pan_name)
            sim_surname = (
                rfdist.JaroWinkler.similarity(passport_name_surname, pan_name)
                if passport_name_surname else 0.0
            )
            sim = max(sim_full, sim_surname)

            if sim < 0.70:
                result, label = "fail",    f"HARD FAIL — {sim:.0%} similarity, below 70% minimum."
            elif sim < 0.85:
                result, label = "warning", f"MANUAL REVIEW — {sim:.0%} (70–85% range)."
            else:
                result, label = "pass",    f"{sim:.0%} similarity, above 85% threshold."

            checks.append({
                "check":  "Name Match — Passport vs PAN",
                "result": result,
                "detail": f"Passport: '{passport_name_full}' · PAN: '{pan_name}' · {label}",
            })
        except Exception as e:
            checks.append({
                "check": "Name Match — Passport vs PAN",
                "result": "warning", "detail": f"Comparison failed: {e}",
            })
    elif passport_name_full or pan_name:
        checks.append({
            "check": "Name Match — Passport vs PAN",
            "result": "warning",
            "detail": "Only one name available — cannot compare.",
        })

    # ── 3: DOB match — passport vs PAN ───────────────────────────────────────
    passport_dob = mrz.get("date_of_birth") or viz.get("date_of_birth_fallback")
    pan_dob      = pan.get("pan_dob")

    if passport_dob and pan_dob:
        try:
            pdob   = date.fromisoformat(passport_dob)
            bandob = date.fromisoformat(pan_dob)
            delta  = abs((pdob - bandob).days)
            if delta > 365:
                # If MRZ not verified (GPT-VIZ-fallback), DOB may be a misread
                # on a low-quality scan — downgrade to warning.
                # Only hard-fail when MRZ check-digit-validated DOB confirms the gap.
                mrz_verified = mrz.get("mrz_verified", False)
                if mrz_verified:
                    checks.append({
                        "check":  "DOB Match — Passport vs PAN",
                        "result": "fail",
                        "detail": (
                            f"DOB gap of {delta} days — passport: {passport_dob}, "
                            f"PAN: {pan_dob}. MRZ checksum-validated — "
                            f"likely different individuals — HARD FAIL."
                        ),
                    })
                else:
                    checks.append({
                        "check":  "DOB Match — Passport vs PAN",
                        "result": "warning",
                        "detail": (
                            f"DOB mismatch: passport {passport_dob} vs PAN {pan_dob} "
                            f"({delta} days gap). Passport DOB was read by GPT on a "
                            f"borderline quality scan — this may be a misread. "
                            f"Verify DOB against the physical passport document."
                        ),
                    })
            elif delta > 30:
                checks.append({
                    "check":  "DOB Match — Passport vs PAN",
                    "result": "warning",
                    "detail": f"DOB gap of {delta} days — verify ({passport_dob} vs {pan_dob}).",
                })
            else:
                checks.append({
                    "check":  "DOB Match — Passport vs PAN",
                    "result": "pass",
                    "detail": f"DOB matches: {passport_dob}.",
                })
        except Exception:
            checks.append({
                "check": "DOB Match — Passport vs PAN",
                "result": "warning",
                "detail": f"Could not compare DOBs: {passport_dob} vs {pan_dob}",
            })
    elif passport_dob or pan_dob:
        checks.append({
            "check": "DOB Match — Passport vs PAN",
            "result": "warning",
            "detail": "Only one DOB available — cannot compare.",
        })

    # ── 4: Passport holder in ticket passenger list (R-TICKET-07) ────────────
    all_passengers = ticket.get("all_passengers", [])
    if passport_name_full and all_passengers:
        ppt_mrz_fmt = None
        if mrz.get("surname") and mrz.get("given_names"):
            ppt_mrz_fmt = f"{mrz['surname']}/{mrz['given_names']}".upper()

        best_match, best_sim = None, 0.0
        for pax in all_passengers:
            pax_name = (pax.get("name") or "").upper()
            sim = max(
                rfdist.JaroWinkler.similarity(passport_name_full, pax_name),
                rfdist.JaroWinkler.similarity(ppt_mrz_fmt or "", pax_name) if ppt_mrz_fmt else 0.0
            )
            if sim > best_sim:
                best_sim, best_match = sim, pax

        if best_sim >= 0.70:
            ticket["passenger_name"] = best_match["name"]
            ticket["pnr"]            = best_match.get("pnr")
            checks.append({
                "check":  "Passport Holder in Ticket",
                "result": "pass",
                "detail": (
                    f"Passport '{passport_name_full}' matched to "
                    f"'{best_match['name']}' ({best_sim:.0%}) — "
                    f"PNR: {best_match.get('pnr') or 'not found'}"
                ),
            })
        else:
            best_name = best_match["name"] if best_match else "none"
            checks.append({
                "check":  "Passport Holder in Ticket",
                "result": "warning",
                "detail": (
                    f"Passport '{passport_name_full}' not found in passenger list. "
                    f"Best match: '{best_name}' ({best_sim:.0%}). Verify manually."
                ),
            })
    elif all_passengers and not passport_name_full:
        # No passport yet — use first passenger as default
        ticket["passenger_name"] = all_passengers[0].get("name")
        ticket["pnr"]            = all_passengers[0].get("pnr")

    # ── 5a: Reload form date vs travel date ──────────────────────────────────
    # The reload form should be dated close to the travel date.
    # A form dated significantly AFTER the travel date (e.g. a year later)
    # means the form may have been re-used, misdated, or is for a different trip.
    # Tolerance: allow up to 90 days before departure (pre-loading is normal)
    # and 7 days after departure (form filled day-of or next day is common).
    reload_form_date_str  = reload.get("date")
    ticket_departure_str  = ticket.get("departure_date")

    if reload_form_date_str and ticket_departure_str:
        try:
            rfd = date.fromisoformat(reload_form_date_str)
            dep = date.fromisoformat(ticket_departure_str)
            delta_days = (rfd - dep).days   # positive = form AFTER departure

            if delta_days > 30:
                checks.append({
                    "check":  "Reload Form Date vs Travel Date",
                    "result": "warning",
                    "detail": (
                        f"Reload form date ({reload_form_date_str}) is {delta_days} days "
                        f"AFTER departure ({ticket_departure_str}). "
                        f"This may be a re-used form or a different transaction — verify with applicant."
                    ),
                })
            elif delta_days < -180:
                checks.append({
                    "check":  "Reload Form Date vs Travel Date",
                    "result": "warning",
                    "detail": (
                        f"Reload form date ({reload_form_date_str}) is {abs(delta_days)} days "
                        f"before departure ({ticket_departure_str}). "
                        f"Unusually early — verify this form is for this trip."
                    ),
                })
            else:
                checks.append({
                    "check":  "Reload Form Date vs Travel Date",
                    "result": "pass",
                    "detail": (
                        f"Reload form dated {reload_form_date_str}, departure {ticket_departure_str} "
                        f"({abs(delta_days)} days {'before' if delta_days <= 0 else 'after'} travel)."
                    ),
                })
        except Exception:
            pass

    # ── 5: Currency vs destination ────────────────────────────────────────────
    # Two sub-checks:
    #  5a: no currency loaded is inappropriate for the destination
    #  5b: the expected currency for that destination is actually loaded
    #      (e.g. going to Spain but loaded only USD is suspicious)

    # Countries where a specific local currency is strongly expected
    _DEST_EXPECTED = {
        "EUR": ["Spain", "France", "Germany", "Italy", "Netherlands", "Portugal",
                "Greece", "Austria", "Belgium", "Finland", "Ireland", "Luxembourg",
                "Malta", "Cyprus", "Slovakia", "Slovenia", "Estonia", "Latvia",
                "Lithuania", "Croatia", "Schengen"],
        "GBP": ["United Kingdom", "UK", "England", "Scotland", "Wales"],
        "JPY": ["Japan"],
        "THB": ["Thailand"],
        "SGD": ["Singapore"],
        "CAD": ["Canada"],
        "CHF": ["Switzerland"],
        "AUD": ["Australia"],
        "AED": ["United Arab Emirates", "UAE", "Dubai", "Abu Dhabi"],
        "SAR": ["Saudi Arabia"],
        "QAR": ["Qatar"],
        "MYR": ["Malaysia"],
        "NZD": ["New Zealand"],
        "HKD": ["Hong Kong"],
    }

    # Currency source priority: annexure (confirmed transaction) >
    # bank transfer (infer from amount + beneficiary) > reload form (customer's
    # handwritten request — least reliable, especially for currency field).
    annexure_cur  = annexure.get("annexure_currency")
    currencies    = (
        [annexure_cur.upper()]              if annexure_cur else
        reload.get("currencies_requested", [])
    )
    currency_source = "annexure" if annexure_cur else "reload form"
    destination = ticket.get("destination_country", "") or ""

    if currencies and destination:
        flags   = []
        missing = []

        # 5a: inappropriate currency?
        for cur in currencies:
            if cur in _UNIVERSAL:
                continue
            if cur in _STRICT:
                if not any(d.lower() in destination.lower() for d in _STRICT[cur]):
                    flags.append(
                        f"{cur} unusual for '{destination}' "
                        f"(typically: {', '.join(_STRICT[cur])})"
                    )

        # 5b: expected currency present?
        for expected_cur, dest_list in _DEST_EXPECTED.items():
            if any(d.lower() in destination.lower() for d in dest_list):
                if expected_cur not in [c.upper() for c in currencies]:
                    missing.append(
                        f"Expected {expected_cur} for '{destination}' "
                        f"but only {currencies} loaded — verify with applicant"
                    )
                break   # only flag the primary expected currency

        all_issues = flags + missing
        checks.append({
            "check":  "Currency vs Destination",
            "result": "warning" if all_issues else "pass",
            "detail": " | ".join(all_issues) if all_issues
                      else f"{currencies} appropriate for '{destination}' (source: {currency_source})",
        })
    elif destination and not currencies:
        checks.append({
            "check": "Currency vs Destination",
            "result": "warning",
            "detail": (
                "Destination is known but no currency could be determined — "
                "upload Annexure or Reload Form to enable this check."
            ),
        })
    elif currencies and not destination:
        checks.append({
            "check": "Currency vs Destination",
            "result": "warning",
            "detail": (
                "Currency is known but ticket destination is missing — "
                "upload Air Ticket to enable this check."
            ),
        })

    # ── 6: SOF statement period vs transfer date ─────────────────────────────
    # The SOF should cover the period when the forex transfer happened.
    # If the SOF ends BEFORE the transfer date, it does not prove the
    # applicant had sufficient funds at the time of transfer — KYC gap.
    sof_from      = sof.get("statement_from")
    sof_to        = sof.get("statement_to")
    transfer_date = transfer.get("transfer_date")

    if sof_from and sof_to and transfer_date:
        try:
            sf = date.fromisoformat(sof_from)
            st = date.fromisoformat(sof_to)
            td = date.fromisoformat(transfer_date)

            if td > st:
                checks.append({
                    "check":  "SOF Period vs Transfer Date",
                    "result": "warning",
                    "detail": (
                        f"Transfer date {transfer_date} is AFTER the SOF statement ends "
                        f"({sof_from} to {sof_to}). Statement does not include the transfer "
                        f"transaction — request a bank statement covering {transfer_date}."
                    ),
                })
            elif td < sf:
                checks.append({
                    "check":  "SOF Period vs Transfer Date",
                    "result": "warning",
                    "detail": (
                        f"Transfer date {transfer_date} is BEFORE the SOF statement begins "
                        f"({sof_from} to {sof_to}). Statement does not cover the transfer period."
                    ),
                })
            else:
                checks.append({
                    "check":  "SOF Period vs Transfer Date",
                    "result": "pass",
                    "detail": (
                        f"Transfer date {transfer_date} falls within SOF period "
                        f"({sof_from} to {sof_to})."
                    ),
                })
        except Exception:
            pass
    elif (sof_from or sof_to) and transfer_date:
        checks.append({
            "check":  "SOF Period vs Transfer Date",
            "result": "warning",
            "detail": "SOF statement period incomplete — cannot verify transfer date coverage.",
        })

    # ── 7: Reload form cardholder name vs passport name ───────────────────────
    # The reload form is filled by the dealer/agent — typos and spelling variants
    # are common (e.g. "Vaidehra" vs "Vadehra"). A significant mismatch may
    # indicate the form was filled for a different applicant.
    reload_name = (
        reload.get("cardholder_name", "").upper().strip()
        if reload.get("cardholder_name") else None
    )

    if passport_name_full and reload_name:
        try:
            sim_r_full    = rfdist.JaroWinkler.similarity(passport_name_full, reload_name)
            sim_r_surname = (
                rfdist.JaroWinkler.similarity(passport_name_surname, reload_name)
                if passport_name_surname else 0.0
            )
            sim_r = max(sim_r_full, sim_r_surname)

            if sim_r < 0.70:
                checks.append({
                    "check":  "Name Match — Passport vs Reload Form",
                    "result": "fail",
                    "detail": (
                        f"Reload form name '{reload_name}' does not match passport "
                        f"'{passport_name_full}' ({sim_r:.0%}) — "
                        f"possible wrong applicant or significant data entry error."
                    ),
                })
            elif sim_r < 0.90:
                checks.append({
                    "check":  "Name Match — Passport vs Reload Form",
                    "result": "warning",
                    "detail": (
                        f"Reload form name '{reload_name}' differs from passport "
                        f"'{passport_name_full}' ({sim_r:.0%}) — "
                        f"likely a spelling variant or typo. Verify with applicant."
                    ),
                })
            else:
                checks.append({
                    "check":  "Name Match — Passport vs Reload Form",
                    "result": "pass",
                    "detail": (
                        f"Reload form name '{reload_name}' matches passport "
                        f"'{passport_name_full}' ({sim_r:.0%})."
                    ),
                })
        except Exception:
            pass

    # ── 8: Bank transfer beneficiary vs applicant name ───────────────────────
    # If the money was sent TO someone other than the applicant or Thomas Cook,
    # the applicant may not be the actual traveller or there is a diversion.
    transfer_beneficiary = (transfer.get("beneficiary") or "").upper().strip()
    _tc_names = {"THOMASCOOK", "THOMAS COOK", "TCIL", "THOMAS COOK INDIA"}

    if transfer_beneficiary and passport_name_full:
        # Skip if beneficiary is Thomas Cook (expected)
        if not any(tc in transfer_beneficiary.replace(" ", "") for tc in
                   {n.replace(" ", "") for n in _tc_names}):
            try:
                sim_ben = rfdist.JaroWinkler.similarity(passport_name_full, transfer_beneficiary)
                if sim_ben < 0.65:
                    checks.append({
                        "check":  "Transfer Beneficiary vs Applicant",
                        "result": "warning",
                        "detail": (
                            f"Transfer beneficiary '{transfer_beneficiary}' does not match "
                            f"applicant '{passport_name_full}' ({sim_ben:.0%}). "
                            f"Verify the money was not sent to a third party."
                        ),
                    })
            except Exception:
                pass

    # ── 9: SOF account holder vs passport name ───────────────────────────────
    # The SOF must belong to the APPLICANT, not the forex dealer or a third party.
    # A dealer submitting their own bank statement as the client's SOF is a
    # KYC failure — the statement must show the applicant's own funds.
    sof_holder = (
        sof.get("account_holder_name", "").upper().strip()
        if sof.get("account_holder_name") else None
    )

    if passport_name_full and sof_holder:
        try:
            sim_sof_full    = rfdist.JaroWinkler.similarity(passport_name_full, sof_holder)
            sim_sof_surname = (
                rfdist.JaroWinkler.similarity(passport_name_surname, sof_holder)
                if passport_name_surname else 0.0
            )
            sim_sof = max(sim_sof_full, sim_sof_surname)

            if sim_sof < 0.70:
                checks.append({
                    "check":  "SOF Account Holder vs Passport",
                    "result": "fail",
                    "detail": (
                        f"SOF account holder '{sof_holder}' does not match passport "
                        f"'{passport_name_full}' ({sim_sof:.0%}) — "
                        f"HARD FAIL: bank statement belongs to a different person. "
                        f"The applicant must submit their own bank statement as Source of Funds."
                    ),
                })
            elif sim_sof < 0.85:
                checks.append({
                    "check":  "SOF Account Holder vs Passport",
                    "result": "warning",
                    "detail": (
                        f"SOF account holder '{sof_holder}' differs from passport "
                        f"'{passport_name_full}' ({sim_sof:.0%}) — "
                        f"verify this is the applicant's own account, not a third party."
                    ),
                })
            else:
                checks.append({
                    "check":  "SOF Account Holder vs Passport",
                    "result": "pass",
                    "detail": (
                        f"SOF account holder '{sof_holder}' matches passport "
                        f"'{passport_name_full}' ({sim_sof:.0%})."
                    ),
                })
        except Exception:
            pass
    elif sof_holder and not passport_name_full:
        pass   # No passport to compare — skip silently

    return checks


# ── Unified field table builder ────────────────────────────────────────────────

def _build_field_table(extracted: dict) -> list:
    """Flatten extracted fields into rows for the unified display table."""
    rows = []

    def add(category, field, value, source, method=None):
        rows.append({
            "Category": category,
            "Field":    field,
            "Value":    str(value).strip() if value not in (None, "", "null") else "—",
            "Source":   source.upper(),
            "Method":   method or "—",
        })

    p = extracted.get("passport", {})
    if p:
        mrz, viz = p.get("mrz", {}), p.get("viz", {})
        base_src = mrz.get("_source", "PassportEye")

        def mrz_method(key):
            return mrz.get(f"{key}_source", base_src)

        add("Identity", "Surname",         mrz.get("surname"),          "passport", mrz_method("surname"))
        add("Identity", "Given Names",     mrz.get("given_names"),      "passport", mrz_method("given_names"))
        add("Identity", "Passport No.",    mrz.get("passport_number"),  "passport", mrz_method("passport_number"))
        add("Identity", "Date of Birth",   mrz.get("date_of_birth"),    "passport", mrz_method("date_of_birth"))
        add("Identity", "Expiry Date",     mrz.get("expiry_date"),      "passport", mrz_method("expiry_date"))
        add("Identity", "Nationality",     mrz.get("nationality"),      "passport", mrz_method("nationality"))
        add("Identity", "Issue Date",      viz.get("issue_date"),       "passport", "GPT-VIZ")
        if viz.get("issue_date_review_flag"):
            add("Identity", "⚠️ Issue Date Flag", viz.get("issue_date_review_reason"), "passport", "review")
        add("Identity", "Place of Issue",  viz.get("place_of_issue"),   "passport", "GPT-VIZ")
        ck = mrz.get("checksum_valid")
        add("Identity", "MRZ Checksum",
            "✅ Valid" if ck is True else ("❌ FAILED" if ck is False else None),
            "passport", base_src)
        add("Identity", "MRZ Verified",
            "✅ PassportEye" if mrz.get("mrz_verified") else "⚠️ GPT-VIZ fallback (PassportEye could not read MRZ)",
            "passport", base_src)

    pan = extracted.get("pan", {})
    if pan:
        add("Identity", "PAN Name",        pan.get("pan_name"),         "pan")
        add("Identity", "PAN Number",      pan.get("pan_number"),       "pan")
        add("Identity", "Father Name",     pan.get("father_name"),      "pan")
        add("Identity", "PAN DOB",         pan.get("pan_dob"),          "pan")
        add("Identity", "Entity Type",     pan.get("entity_type"),      "pan")

    visa = extracted.get("visa", {})
    if visa:
        add("Travel",  "Visa Number",      visa.get("visa_number"),       "visa")
        add("Travel",  "Destination",      visa.get("destination_country"), "visa")
        add("Travel",  "Visa Issue Date",  visa.get("issue_date"),        "visa")
        add("Travel",  "Visa Expiry",      visa.get("expiry_date"),       "visa")
        add("Travel",  "Visa Type",        visa.get("visa_type"),         "visa")
        add("Travel",  "Issuing Authority", visa.get("issuing_authority"), "visa")

    ticket = extracted.get("ticket", {})
    if ticket:
        add("Travel",  "Passenger Name",   ticket.get("passenger_name"),   "ticket")
        add("Travel",  "PNR",              ticket.get("pnr"),              "ticket")
        add("Travel",  "Airline",          ticket.get("airline"),          "ticket")
        add("Travel",  "Departure Date",
            ticket.get("departure_date") or ticket.get("departure_date_raw"), "ticket")
        add("Travel",  "Return Date",
            ticket.get("return_date") or ticket.get("return_date_raw"),      "ticket")
        add("Travel",  "Origin IATA",      ticket.get("origin_iata"),      "ticket")
        add("Travel",  "Destination IATA", ticket.get("destination_iata"), "ticket")
        add("Travel",  "Destination Country", ticket.get("destination_country"), "ticket")
        legs = ticket.get("legs", [])
        if legs:
            add("Travel", "Legs Parsed", f"{len(legs)} flight legs extracted", "ticket")

    reload = extracted.get("reload", {})
    if reload:
        curs    = reload.get("currencies_requested", [])
        amt_map = reload.get("currency_amounts", {})
        if amt_map:
            curs_display = ", ".join(
                f"{k} {v}" for k, v in amt_map.items()
                if v and str(v).strip() not in ("", "null", "0")
            )
        else:
            curs_display = ", ".join(curs) if curs else None

        # Identity cross-validation rows from the reload form
        add("Identity", "Reload Name",        reload.get("cardholder_name"),   "reload")
        add("Identity", "Reload Passport No.", reload.get("passport_number"),  "reload")
        add("Identity", "Reload DOB",         reload.get("date_of_birth"),     "reload")
        add("Identity", "Reload PAN",         reload.get("pan_number"),        "reload")

        # Financial fields
        add("Financial", "Cardholder Name",   reload.get("cardholder_name"),   "reload")
        add("Financial", "Currencies",        curs_display,                    "reload")
        add("Financial", "Card Type",         reload.get("card_type"),         "reload")
        add("Financial", "Card Network",      reload.get("card_network"),      "reload")
        add("Financial", "Purpose of Travel", reload.get("purpose_of_travel"),"reload")
        add("Financial", "Form Date",         reload.get("date"),              "reload")

    transfer = extracted.get("transfer", {})
    if transfer:
        add("Financial", "Transfer Amount",   transfer.get("amount_inr"),       "transfer")
        add("Financial", "Reference / UTR",   transfer.get("reference_number"), "transfer")
        add("Financial", "Transfer Date",     transfer.get("transfer_date"),    "transfer")
        add("Financial", "Beneficiary",       transfer.get("beneficiary"),      "transfer")

    annexure = extracted.get("annexure", {})
    if annexure:
        add("Financial", "Dealer Name",       annexure.get("dealer_name"),      "annexure")
        add("Financial", "RBI Licence",       annexure.get("rbi_licence"),      "annexure")
        add("Financial", "Annexure Currency", annexure.get("currency"),         "annexure")
        add("Financial", "Annexure Amount",   annexure.get("amount"),           "annexure")
        add("Financial", "Annexure UTR",      annexure.get("utr_reference"),    "annexure")

    sof = extracted.get("sof", {})
    if sof:
        period = None
        if sof.get("statement_from") and sof.get("statement_to"):
            period = f"{sof['statement_from']} to {sof['statement_to']}"
        add("Financial", "Account Holder",    sof.get("account_holder_name"),  "sof")
        add("Financial", "Bank",              sof.get("bank_name"),            "sof")
        add("Financial", "Statement Period",  period,                          "sof")
        add("Financial", "Closing Balance",   sof.get("closing_balance"),      "sof")

    return rows


# ── Minimum viable fields (R-DIQA-04) ─────────────────────────────────────────
# Rejection is field-level, not image-level.

MINIMUM_VIABLE_FIELDS = {
    "passport": ["surname", "given_names", "passport_number", "date_of_birth", "expiry_date"],
    "pan":      ["pan_number", "pan_name", "pan_dob"],
    "visa":     ["visa_number", "issue_date", "expiry_date"],
    "ticket":   ["departure_date_raw", "origin_iata", "destination_iata"],
    "reload":   ["cardholder_name", "currencies_requested"],
    "transfer": ["amount_inr", "reference_number"],
    "annexure": ["dealer_name", "cardholder_name"],
    "sof":      ["account_holder_name", "closing_balance"],
}

_FIELD_REMEDIATION = {
    "surname":          "Passport surname not readable — ensure bio-data page is fully visible and photo taken straight-on",
    "given_names":      "Given name not readable — ensure bio-data page is fully visible",
    "passport_number":  "Passport number not readable — ensure top-right of bio-data page is in frame",
    "date_of_birth":    "Date of birth not readable — ensure bio-data page is unobstructed",
    "expiry_date":      "Expiry date not readable — check MRZ strip at bottom is visible and horizontal",
    "pan_number":       "PAN number not readable — photograph the card directly, avoid glare",
    "pan_name":         "Name on PAN not readable — ensure card is flat and well-lit",
    "pan_dob":          "Date of birth on PAN not readable",
    "visa_number":      "Visa number not readable — ensure the visa sticker/document is fully in frame",
    "issue_date":       "Visa issue date not readable",
    "departure_date_raw": "Departure date not readable — ensure the flight date is visible",
    "origin_iata":      "Departure airport not found — ensure the itinerary page is included",
    "destination_iata": "Destination airport not found — ensure the full itinerary is included",
    "cardholder_name":  "Cardholder name not readable",
    "currencies_requested": "Currency table not found — ensure page 2 of the reload form is included",
    "amount_inr":       "Transfer amount not readable",
    "reference_number": "Reference/UTR number not readable",
    "dealer_name":      "Dealer name not readable — ensure the letterhead is in frame",
    "account_holder_name": "Account holder name not readable",
    "closing_balance":  "Closing balance not readable — ensure the statement summary section is in frame",
}


def check_minimum_viable_fields(doc_id: str, extracted: dict) -> dict:
    """
    Check whether minimum viable fields are present in extracted data.
    R-DIQA-04: rejection is field-level, not image-level.
    """
    required = MINIMUM_VIABLE_FIELDS.get(doc_id, [])
    if not required:
        return {"viable": True, "missing_fields": [], "messages": []}

    def _val(v):
        return v is not None and str(v).strip() not in ("", "null", "—", "None", "[]")

    if doc_id == "passport":
        mrz = extracted.get("mrz", {})
        field_vals = {f: mrz.get(f) for f in required}
    else:
        field_vals = {f: extracted.get(f) for f in required}

    missing = [f for f, v in field_vals.items() if not _val(v)]

    return {
        "viable":         len(missing) == 0,
        "missing_fields": missing,
        "messages":       [_FIELD_REMEDIATION.get(f, f"'{f}' could not be extracted") for f in missing],
    }


# ── Cross-validation entry point ───────────────────────────────────────────────

def run_cross_validation(extracted_by_doc: dict) -> dict:
    """
    Run cross-document validation using pre-extracted field data.
    Called by diqa_test.py after per-document extraction at upload time.

    IMPORTANT: _cross_validate runs BEFORE _build_field_table.
    _cross_validate may update ticket["passenger_name"] and ticket["pnr"]
    in-place via passport-name fuzzy match. The field table must be built
    AFTER this so it reflects the updated values.
    """
    cross_validation = _cross_validate(extracted_by_doc)
    field_table      = _build_field_table(extracted_by_doc)
    return {
        "field_table":      field_table,
        "cross_validation": cross_validation,
    }


# ── Batch entry point ──────────────────────────────────────────────────────────

def run(usable_docs: dict, progress_callback=None) -> dict:
    """
    Batch entry point — called from diqa_test.py.

    Args:
        usable_docs       : {doc_id: {"file_bytes": bytes, "filetype": str, ...}}
        progress_callback : optional callable(doc_id, step)

    Returns:
        {"extracted": dict, "field_table": list, "cross_validation": list}
    """
    extracted = {}

    for doc_id, state in usable_docs.items():
        if progress_callback:
            progress_callback(doc_id, "starting")
        fn = _EXTRACTORS.get(doc_id)
        if fn:
            try:
                extracted[doc_id] = fn(state["file_bytes"], state["filetype"])
            except Exception as e:
                extracted[doc_id] = {"source": doc_id, "error": str(e)}
            finally:
                if progress_callback:
                    progress_callback(doc_id, "done")

    return {
        "extracted":          extracted,
        **run_cross_validation(extracted),
    }