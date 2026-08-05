"""
Document Image Quality Assessment (DIQA) Module - v2
-----------------------------------------------------
Tiered Hybrid Architecture:

Tier 1A — Heuristics (blur, exposure)
         OpenCV only, runs in milliseconds
         Catches obvious failures immediately

Tier 1B — CG-DIQA (Character Gradient)
         MSER + Sobel via OpenCV, runs in milliseconds
         Replaces the flawed global std-dev contrast check
         Immune to sparse form whitespace dilution

Tier 2  — GPT-4o-mini Vision with Chain-of-Thought
         API fallback, only triggered when CG-DIQA is borderline
         or MSER collapses on heavily degraded phone photos

Pattern A: Pure quality gate — returns pass/fail verdict only.
Does NOT extract text. That is left to the downstream OCR engine.

Usage:
    from diqa import assess
    result = assess(file_bytes, filetype="pdf")
"""

import os
import cv2
import json
import base64
import numpy as np
import fitz
from openai import OpenAI

# ── Thresholds ─────────────────────────────────────────────────────────────────
BLUR_THRESHOLD        = 80.0        # Variance of Laplacian. Below = out of focus.
BRIGHTNESS_LOW        = 45.0        # Mean pixel value. Below = too dark.
PDF_RENDER_DPI        = 150         # DPI for rendering PDF pages to images.
MAX_FILE_SIZE_BYTES   = 2_097_152   # 2MB hard limit. Reject before any processing.

# CG-DIQA routing thresholds
CGDIQA_PASS           = 15.0   # Score above this → confident enough to assess
CGDIQA_FAIL           = 5.0    # Score below this (regions found) → confident FAIL
CGDIQA_REVIEW         = 130.0  # Score between PASS and REVIEW → pass but flag for human review
                                # Based on empirical data: Schengen sticker visa scored 96.6,
                                # photocopy passports ~165-175. Anything below 130 is borderline.
CGDIQA_MIN_REGIONS    = 10     # Fewer valid MSER regions than this → route to API
MSER_ASPECT_MIN       = 0.25   # Discard MSER boxes with width/height ratio below this
MSER_ASPECT_MAX       = 4.0    # Discard MSER boxes with width/height ratio above this


# ── Document type classification signals ───────────────────────────────────────
# Top 3 confirmation signals per document type, derived from manual research
# across 8 forex/travel-card pipeline document types.
# Each entry drives the GPT-4o-mini classification prompt in _check_doc_type().

_DOC_SIGNALS = {
    "passport": {
        "name": "Indian Passport (bio-data page)",
        "signals": [
            "MRZ at the bottom — two lines of OCR-B text, first line starting with 'P<IND'",
            "Passport number in format: 1 letter + 7 digits (e.g. Z3466012, U2804406)",
            "'REPUBLIC OF INDIA' / 'भारत गणराज्य' header and bilingual Hindi-English field labels",
        ],
        "not_this": "Aadhaar card, PAN card, driving licence, voter ID, visa/entry permit, selfie",
        "prompt_override": """You are classifying a document image to verify it is an Indian passport bio-data page.

IMPORTANT — ORIENTATION: The image may be rotated 0°, 90°, 180°, or 270°.
Do not reject a document based on orientation alone. Mentally rotate and
check all four orientations before deciding.

IMPORTANT — SCAN FORMAT: The image may show a single page or a two-page
passport spread (bio-data page alongside the address/family details page).
Both formats are valid passport submissions.

Check for these 3 signals in ANY orientation:

SIGNAL 1 — MRZ (strongest signal):
Look for two lines of machine-readable text containing '<' fill characters
and the country code 'IND'. The first line starts with 'P<IND'. These lines
may appear horizontal, vertical, or diagonal depending on rotation.
Presence of '<<<<' fill characters anywhere in the image is a strong
indicator of an MRZ even if you cannot read the full line.

SIGNAL 2 — Passport number:
Look for a standalone alphanumeric code in the format:
one uppercase letter followed by exactly 7 digits (e.g. Z#######, A#######).
This appears in the top-right area of the bio-data section when correctly oriented.

SIGNAL 3 — Header and bilingual labels:
Look for 'REPUBLIC OF INDIA' and/or its Hindi equivalent 'भारत गणराज्य'
in any orientation. Also look for bilingual field labels (English + Hindi
script side by side) for fields like Surname, Nationality, Date of Birth.

SECONDARY SIGNALS (use if primary signals are unclear due to rotation or partial scan):
- A passport-style photo embedded in the document
- The address page showing: father's name, mother's name, spouse name,
  full address in all-caps — this confirms a passport bundle even if the
  bio-data page is unclear
- 'ORIGINAL SEEN & VERIFIED' or 'ORIGINAL SEEN AND VERIFIED' stamp

DECISION RULE:
- MRZ with 'P<IND' found in any orientation → PASSPORT, accept
- '<<<<' fill characters + 'IND' found + bilingual labels → PASSPORT, accept
- Photo + 'REPUBLIC OF INDIA' in any orientation → PASSPORT, accept
- None of the above found after checking all orientations → reject

DO NOT classify as ID card based on layout alone. A horizontally narrow
strip containing a photo and data fields is the correct appearance of a
passport bio-data page when the full scan is rotated 90°.

Output ONLY this JSON with no other text:
{"match": true, "detected_type": "passport", "reason": "confirmed — [which signals found]"}
or
{"match": false, "detected_type": "one of: passport|pan|visa|reload|transfer|ticket|annexure|sof|unknown", "reason": "No passport signals detected — [what this actually appears to be]"}"""
    },
    "pan": {
        "name": "PAN Card (individual Permanent Account Number)",
        "signals": [
            "PAN number format: 10 characters — 5 letters, 4 digits, 1 letter (e.g. AFHPG2521G)",
            "'PERMANENT ACCOUNT NUMBER' / 'स्थायी लेखा संख्या' header (any case)",
            "'FATHER'S NAME' / 'पिता का नाम' field — unique to individual PAN cards",
        ],
        "not_this": "Aadhaar, passport, voter ID, driving licence, bank card, debit card",
        "prompt_override": """You are classifying a document image to verify it is an individual's PAN card.

IMPORTANT — CARD SIZE: The PAN card may appear as a small embedded rectangle 
within a larger scanned page. Focus on the rectangular card area only — 
do not assess the surrounding white space, stamps, or signatures.

SIGNAL 1 — PAN number (strongest signal):
Look for a 10-character alphanumeric code on the card face: exactly 5 letters, 
4 digits, 1 letter (e.g. AFHPG2521G, ABCDE1234F).
The 4th character indicates the PAN holder type:
  P = Individual person → VALID for this slot
  F, C, H, A, B, G, J, L, T = Company/Firm/Entity → REJECT as company PAN

SIGNAL 2 — Header text:
Look for 'Permanent Account Number' or 'PERMANENT ACCOUNT NUMBER' or 
'स्थायी लेखा संख्या' in any case. Also accept 'Permanent A/c No.'
This text appears on the card face regardless of individual or company PAN.

SIGNAL 3 — Individual vs company distinction:
  Individual PAN: shows 'FATHER'S NAME' / 'पिता का नाम' field + cardholder photo
  Company PAN: shows entity name + registration date, NO father's name, NO photo

DECISION RULES:
- Signal 1 present AND 4th character is P AND Signal 3 shows individual fields 
  → INDIVIDUAL PAN, accept
- Signal 1 present AND 4th character is F/C/H/A/B/G/J/L/T 
  → COMPANY PAN, reject with specific message
- Signal 2 present but no PAN number readable (small card, low resolution)
  → accept if card layout matches individual PAN format
- No PAN signals found at all → reject

Output ONLY this JSON with no other text:
{"match": true, "detected_type": "pan", "reason": "confirmed — individual PAN, [signals found]"}
or
{"match": false, "detected_type": "pan", "reason": "Company/Firm PAN detected (entity PAN, not individual PAN) — this slot requires the individual cardholder's personal PAN card"}
or
{"match": false, "detected_type": "one of: passport|visa|reload|transfer|ticket|annexure|sof|unknown", "reason": "wrong document — [what this actually appears to be]"}"""
    },
    "visa": {
        "name": "Visa (e-Visa or sticker visa)",
        "signals": [
            "'Visa issue date:' AND 'Visa valid till:' fields present together on the same document",
            "Country immigration authority branding (e.g. ICA Singapore, UAE Federal Authority, GDRFA)",
            "'Type of Visa:' field with value such as TOURIST, MULTIPLE JOURNEY, or SINGLE ENTRY",
        ],
        "not_this": "Passport bio-data page (has MRZ instead), flight ticket, boarding pass, hotel booking",
    },
    "reload": {
        "name": "Thomas Cook Reload / Application Form",
        "signals": [
            "'Thomas Cook Travel Cards' header AND 'A FAIRFAX Company' text on same document",
            "Currency selection table listing USD, GBP, EUR, AUD, CAD, CHF, JPY, SGD, AED, THB with Amount column",
            "'travelcards@thomascook.in' email address printed on the form",
        ],
        "not_this": "Dealer annexure letter (has RBI licence), bank transfer, generic forex receipt",
        "prompt_override": """You are verifying that a document image is a Thomas Cook Travel Card reload or application form.

IMPORTANT — PAGE STRUCTURE: This form is always multi-page.
  Page 1: Thomas Cook branding, 'A FAIRFAX Company', Cardholder Declaration, Important Information
  Page 2: 10-currency selection table, Fees and Limits, personal details, customer signature
You will receive both pages — assess ALL pages before deciding.

SIGNAL 1 — Thomas Cook branding (Page 1):
Look for 'Thomas Cook | Travel Cards' or 'Thomas Cook Travel Cards' header AND
'A FAIRFAX Company' text on the same page. Both must be present together.

SIGNAL 2 — Currency table (Page 2):
Look for a table listing these currencies with an Amount column:
USD (US Dollar), GBP (GB Pound), EUR (Euro), AUD (Australian Dollar),
CAD (Canadian Dollar), CHF (Swiss Franc), JPY (Japanese Yen),
SGD (Singapore Dollar), AED (UAE Dirham), THB (Thai Baht).
This table is always on the second page.

SIGNAL 3 — Form section headers:
Look for ANY of: 'Cardholder Declaration', 'Borderless Prepaid Card',
'One Currency Card', 'Additional Declaration for Business Card Application'.
These confirm this is Thomas Cook's own form, not a dealer letter.

DECISION RULE:
- Signal 1 (branding) found on any page → strong indicator
- Signal 2 (currency table) found on any page → strong indicator
- If EITHER Signal 1 OR Signal 2 is found, and Signal 3 supports it → ACCEPT
- Do not reject solely because the currency table is missing from Page 1
- Do not reject solely because the email 'travelcards@thomascook.in' is not visible — it appears in small body text

Output ONLY this JSON with no other text:
{"match": true, "detected_type": "reload", "reason": "confirmed — [which signals found on which pages]"}
or
{"match": false, "detected_type": "one of: passport|pan|visa|transfer|ticket|annexure|sof|unknown", "reason": "wrong document — [what this actually appears to be]"}"""
    },
    "transfer": {
        "name": "Bank Transfer Acknowledgement (online netbanking)",
        "signals": [
            "3-step progress UI: ENTER DETAILS → CONFIRM TRANSACTION → ACKNOWLEDGEMENT with Step 3 active",
            "'Processing Successful.' status with a single UTR/Reference number and INR amount",
            "Bank netbanking portal URL visible (e.g. netbanking.hdfcbank.com, onlinebanking.axisbank.co.in)",
        ],
        "not_this": "Multi-row bank statement, UPI payment screenshot, cheque image, demand draft",
    },
    "ticket": {
        "name": "Air Ticket / Flight Itinerary",
        "signals": [
            "'BOOKING REFERENCE:' with a 6-character alphanumeric PNR code (e.g. RVIFPT, 6QNJLB)",
            "Flight number in 2-letter IATA airline code + digits format (e.g. SQ401, TK4650, 6E123)",
            "IATA airport codes with route arrow (e.g. DEL → SIN) and departure/arrival times",
        ],
        "not_this": "Visa/entry permit, hotel booking confirmation, boarding pass, bus/train ticket",
    },
    "annexure": {
        "name": "Annexure / Dealer Declaration Letter",
        "signals": [
            "RBI Authorised Dealer Licence number in DEL.FFMC/XXX/YYYY format on dealer letterhead",
            "'has to reload Thomas Cook Card' phrase in letter body alongside a customer details table",
            "Thomas Cook India payee bank account number (57500000424836) with UTR/reference",
        ],
        "not_this": "Thomas Cook's own reload form (has currency table), bank statement, general business letter",
    },
    "sof": {
        "name": "Source of Funds — Bank Account Statement",
        "signals": [
            "'STATEMENT SUMMARY' section listing Opening Balance, Closing Balance, Dr Count, Cr Count",
            "Multi-row transaction table with columns: Date, Narration/Description, Withdrawal, Deposit, Closing Balance",
            "'This is a computer generated statement and does not require signature' disclaimer",
        ],
        "not_this": "Single-transaction bank transfer confirmation (3-step UI), salary slip, credit card statement",
    },
}

# Pairs where a mismatch triggers a soft warning (human review) rather than hard reject.
# These are genuinely similar document types where GPT-4o-mini or the user could reasonably
# confuse them, especially on partial scans.
_SOFT_MISMATCH_PAIRS = {
    frozenset({"annexure", "reload"}),   # Both mention Thomas Cook — letterhead vs currency table
    frozenset({"transfer", "sof"}),      # Both are bank documents — single row vs multi-row statement
}


# ── OpenAI client (lazy init) ──────────────────────────────────────────────────
_openai_client = None

def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found in environment. "
                "Add it to your .env file or set it in the shell."
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


# ── PDF handling ───────────────────────────────────────────────────────────────

def _pdf_to_images(pdf_bytes: bytes) -> list:
    """Convert PDF bytes into a list of OpenCV BGR images, one per page."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = PDF_RENDER_DPI / 72
    mat = fitz.Matrix(zoom, zoom)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        arr = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
    doc.close()
    return images


# ── Tier 1A: Heuristic checks ──────────────────────────────────────────────────

def _check_blur(gray: np.ndarray):
    """
    Variance of Laplacian — low variance = missing edge data.

    Distinguishes two near-zero sharpness cases:
    - Blank page: near-zero sharpness AND very high brightness (uniform white,
      no edges to measure). Users need to know page is blank, not blurry.
    - Genuine blur: near-zero sharpness regardless of brightness.
    Both fail — but with different messages so the user knows what to fix.
    """
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if score < BLUR_THRESHOLD:
        mean = float(gray.mean())
        if score < 10.0 and mean > 235:
            return False, (
                f"Page appears blank or near-blank — no text content detected "
                f"(sharpness: {score:.1f})"
            ), score
        return False, (
            f"Image is out of focus — retake from a steady position "
            f"(sharpness: {score:.1f}, minimum: {BLUR_THRESHOLD})"
        ), score
    return True, None, score


def _check_exposure(gray: np.ndarray):
    """
    Catches underexposure only.

    The previous combined check (high brightness + low global contrast → FAIL)
    caused two confirmed false negatives on pages with large blank areas —
    blank visa pages and sparse legend pages inflate global brightness and
    collapse global std dev even when the actual text is perfectly readable.

    CG-DIQA handles overexposure correctly: if text is genuinely washed out,
    MSER finds no character regions and the gradient score collapses naturally.
    We no longer need a global pre-filter for overexposure here.
    """
    mean = float(gray.mean())

    if mean < BRIGHTNESS_LOW:
        return False, (
            f"Image is too dark — improve lighting "
            f"(brightness: {mean:.1f}, minimum: {BRIGHTNESS_LOW})"
        ), mean

    return True, None, mean


# ── Tier 1B: CG-DIQA ──────────────────────────────────────────────────────────

def _check_cgdiqa(gray: np.ndarray) -> tuple:
    """
    Character Gradient based DIQA.

    Steps:
    1. MSER detects stable text character regions (ignores white space natively)
    2. Aspect ratio filter removes borders, underlines, and noise blobs
    3. Sobel gradient measured exclusively within valid character patches
    4. Std dev of those gradients = quality score

    Returns: (score, n_regions, status)
    status: "pass" | "fail" | "borderline"

    Borderline triggers Tier 2 API assessment.
    """
    mser = cv2.MSER_create()

    try:
        _, bboxes = mser.detectRegions(gray)
    except Exception:
        return 0.0, 0, "borderline"

    if bboxes is None or len(bboxes) == 0:
        return 0.0, 0, "borderline"

    # Filter by aspect ratio — removes page borders, long underlines, sensor noise
    valid_bboxes = [
        (x, y, w, h) for (x, y, w, h) in bboxes
        if h > 0 and MSER_ASPECT_MIN < (w / h) < MSER_ASPECT_MAX
    ]
    n_regions = len(valid_bboxes)

    # Too few regions — MSER couldn't find enough text, route to API
    if n_regions < CGDIQA_MIN_REGIONS:
        return 0.0, n_regions, "borderline"

    # Build mask from valid bounding boxes
    mask = np.zeros_like(gray)
    for (x, y, w, h) in valid_bboxes:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # Sobel gradient magnitude
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)

    # Measure only within character regions
    masked = magnitude[mask > 0]
    if len(masked) == 0:
        return 0.0, n_regions, "borderline"

    score = float(np.std(masked))

    if score >= CGDIQA_REVIEW:
        return score, n_regions, "pass"
    elif score >= CGDIQA_PASS:
        return score, n_regions, "review"   # borderline quality — flag for human review
    elif score < CGDIQA_FAIL:
        return score, n_regions, "fail"
    else:
        return score, n_regions, "borderline"  # routes to Tier 2 API


# ── Tier 2: GPT-4o-mini Vision ────────────────────────────────────────────────

_COT_PROMPT = """You are evaluating whether text in this document image can be reliably extracted by an OCR engine.

Analyze each dimension in order before deciding:
1. TEXT SHARPNESS: Are character edges sharp enough to distinguish individual letters? Look for blur or motion artifacts on the text strokes themselves.
2. CONTRAST: Is there sufficient contrast between text ink and background? Look for glare, washout, or shadow obscuring the text.
3. OBSTRUCTION: Is any significant portion of text physically obscured by stains, tears, or damage?
4. COVERAGE: Is enough of the document readable for meaningful extraction?

Note: Empty white space and blank form fields are normal document features — do not penalise a sparse form for having white space.

After analyzing each dimension, output ONLY this JSON with no other text:
{"pass": true, "reason": "one specific sentence", "failed_check": "none"}
or
{"pass": false, "reason": "one specific sentence", "failed_check": "blur" or "contrast" or "obstruction" or "coverage"}"""


def _tier2_api(img: np.ndarray) -> dict:
    """
    Tier 2: GPT-4o-mini with Chain-of-Thought prompt.
    Only called when CG-DIQA is inconclusive or MSER collapses.
    Encodes image as base64 JPEG and sends to OpenAI vision API.
    """
    try:
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf).decode("utf-8")

        client = _get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "text",
                            "text": _COT_PROMPT
                        }
                    ]
                }
            ],
            max_tokens=150,
            temperature=0
        )

        raw = response.choices[0].message.content.strip()
        clean = raw.replace("```json", "").replace("```", "").strip()
        verdict = json.loads(clean)
        verdict["tier"] = 2
        return verdict

    except Exception as e:
        # Fail safe — if API is unavailable, do not silently pass the document
        return {
            "pass": False,
            "reason": f"Tier 2 quality assessment unavailable ({type(e).__name__}: {e}). "
                      f"Please check your OPENAI_API_KEY and network connection.",
            "failed_check": "api_error",
            "tier": 2
        }


# ── Core assessment logic ──────────────────────────────────────────────────────

def _assess_single_image(img: np.ndarray) -> dict:
    """
    Full tiered assessment for a single image.
    Returns a result dict with pass/fail, reason, tier, and scores.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scores = {}

    # ── Tier 1A: Heuristics ────────────────────────────────────────────────────
    passed, reason, scores["blur"] = _check_blur(gray)
    if not passed:
        return {
            "pass": False, "reason": reason,
            "failed_check": "blur", "scores": scores, "tier": 1
        }

    passed, reason, scores["brightness"] = _check_exposure(gray)
    if not passed:
        return {
            "pass": False, "reason": reason,
            "failed_check": "exposure", "scores": scores, "tier": 1
        }

    # ── Tier 1B: CG-DIQA ───────────────────────────────────────────────────────
    cgdiqa_score, n_regions, status = _check_cgdiqa(gray)
    scores["cgdiqa"]       = round(cgdiqa_score, 2)
    scores["mser_regions"] = n_regions

    if status == "pass":
        return {
            "pass": True,
            "flagged_for_review": False,
            "reason": "Passes all quality checks",
            "failed_check": None,
            "scores": scores,
            "tier": 1
        }

    if status == "review":
        return {
            "pass": True,
            "flagged_for_review": True,
            "reason": (
                f"Document quality is borderline — security patterns, photograph noise, or "
                f"photocopy quality may reduce OCR accuracy "
                f"(CG-DIQA score: {cgdiqa_score:.1f}, clean threshold: {CGDIQA_REVIEW}). "
                f"Recommend human review before processing."
            ),
            "failed_check": None,
            "scores": scores,
            "tier": 1
        }

    if status == "fail":
        return {
            "pass": False,
            "reason": (
                f"Text quality insufficient for OCR — character gradients too weak "
                f"(CG-DIQA score: {cgdiqa_score:.1f}, minimum: {CGDIQA_PASS})"
            ),
            "failed_check": "readability",
            "scores": scores,
            "tier": 1
        }

    # ── Tier 2: API fallback for borderline cases ──────────────────────────────
    verdict = _tier2_api(img)
    verdict["scores"] = scores
    return verdict


# ── Document type validation ───────────────────────────────────────────────────

def _check_doc_type(images: list, expected_type: str) -> dict:
    """
    Validates that the uploaded document matches the declared type.
    Accepts the full list of page images — sends up to 3 pages to
    GPT-4o-mini in a single API call so signals split across pages
    (e.g. reload form branding on P1, currency table on P2) are visible.

    Returns:
        {
            "match"         : bool
            "detected_type" : str
            "severity"      : "hard" | "soft" | None
            "reason"        : str
        }

    Severity:
        "hard" — clear mismatch between unrelated document categories.
        "soft" — mismatch between similar types (annexure/reload, transfer/sof).
        None   — match confirmed or check skipped.
    """
    if expected_type not in _DOC_SIGNALS:
        return {"match": True, "detected_type": None, "severity": None,
                "reason": f"Unknown type '{expected_type}' — validation skipped"}

    spec = _DOC_SIGNALS[expected_type]
    valid_types = "|".join(_DOC_SIGNALS.keys())

    if "prompt_override" in spec:
        prompt = spec["prompt_override"]
    else:
        signals_text = "\n".join(
            f"  {i + 1}. {s}" for i, s in enumerate(spec["signals"])
        )
        prompt = f"""You are verifying that a document image matches a specific expected document type.

Expected document type: {spec["name"]}

This document MUST show all three of these signals to be confirmed correct:
{signals_text}

These are examples of WRONG uploads for this slot: {spec["not_this"]}

You may receive multiple page images — assess ALL pages before deciding.

Examine all images carefully. Output ONLY this JSON with no other text:
{{"match": true, "detected_type": "{expected_type}", "reason": "confirmed — brief note on which signals you found"}}
or
{{"match": false, "detected_type": "one of: {valid_types} or unknown", "reason": "wrong document — [what this actually appears to be] — expected {spec['name']}"}}"""

    try:
        # Build content array — send up to 3 pages in one API call
        # This ensures signals split across pages (e.g. reload form) are all visible
        content = []
        for img in images[:3]:
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buf).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high"
                }
            })
        content.append({"type": "text", "text": prompt})

        client = _get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}],
            max_tokens=120,
            temperature=0
        )

        raw = response.choices[0].message.content.strip()
        clean = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)

        # Determine severity for mismatches
        if not result.get("match", True):
            detected = result.get("detected_type", "unknown")
            pair = frozenset({expected_type, detected})
            result["severity"] = "soft" if pair in _SOFT_MISMATCH_PAIRS else "hard"
        else:
            result["severity"] = None

        return result

    except Exception as e:
        return {
            "match": True,
            "detected_type": None,
            "severity": None,
            "reason": f"Document type check unavailable ({type(e).__name__}) — skipped"
        }


# ── Public API ─────────────────────────────────────────────────────────────────

def assess(file_bytes: bytes, filetype: str, expected_doc_type: str = None) -> dict:
    """
    Main entry point for the DIQA module.

    Args:
        file_bytes        : Raw bytes of the uploaded file.
        filetype          : "pdf" or "image"
        expected_doc_type : Optional. One of: passport, pan, visa, reload,
                            transfer, ticket, annexure, sof.
                            If provided, runs document type validation via
                            GPT-4o-mini after quality checks pass.
                            Mismatch is a hard reject.

    Returns:
        {
            "pass"           : bool
            "reason"         : str
            "failed_check"   : str | None
            "failed_pages"   : list[int]
            "page_results"   : list[dict]
            "doc_type_check" : dict | None   (present when expected_doc_type given)
        }
    """
    if filetype == "pdf":
        # ── File size gate (runs before anything else) ─────────────────────────
        size_mb = len(file_bytes) / 1_048_576
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            return {
                "pass": False,
                "reason": (
                    f"File size {size_mb:.1f}MB exceeds the 2MB limit — "
                    f"compress the file or re-scan at a lower resolution before uploading."
                ),
                "failed_check": "file_size",
                "failed_pages": [],
                "page_results": []
            }

        images = _pdf_to_images(file_bytes)
        if not images:
            return {
                "pass": False,
                "reason": "Could not render any pages from the PDF.",
                "failed_pages": [],
                "page_results": []
            }
    else:
        size_mb = len(file_bytes) / 1_048_576
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            return {
                "pass": False,
                "reason": (
                    f"File size {size_mb:.1f}MB exceeds the 2MB limit — "
                    f"compress the image before uploading."
                ),
                "failed_check": "file_size",
                "failed_pages": [],
                "page_results": []
            }
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {
                "pass": False,
                "reason": "Could not decode the image file — it may be corrupted.",
                "failed_pages": [],
                "page_results": []
            }
        images = [img]

    page_results = []
    for i, img in enumerate(images):
        result = _assess_single_image(img)
        result["page"] = i + 1
        page_results.append(result)

    failed = [r for r in page_results if not r["pass"]]
    flagged = [r for r in page_results if r.get("flagged_for_review")]
    overall_pass = len(failed) == 0

    # ── Document type validation (only if quality passed and type was declared) ─
    doc_type_check = None
    if overall_pass and expected_doc_type:
        doc_type_check = _check_doc_type(images, expected_doc_type)
        if not doc_type_check["match"]:
            severity = doc_type_check.get("severity", "hard")
            return {
                "pass": False,
                "requires_review": severity == "soft",
                "reason": doc_type_check["reason"],
                "failed_check": "doc_type",
                "failed_pages": [],
                "page_results": page_results,
                "doc_type_check": doc_type_check,
            }

    return {
        "pass": overall_pass,
        "requires_review": bool(flagged) if overall_pass else False,
        "flagged_pages": [r["page"] for r in flagged],
        "reason": (
            "All pages pass quality checks"
            if overall_pass and not flagged
            else f"{len(flagged)} page(s) flagged for human review — borderline quality"
            if overall_pass and flagged
            else f"{len(failed)} of {len(page_results)} page(s) failed quality checks"
        ),
        "failed_pages": [r["page"] for r in failed],
        "page_results": page_results,
        "doc_type_check": doc_type_check,
    }