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
# Blur thresholds — sourced from production DIQA research
# Dynamsoft production framework (ref [9]): 200.0 for 300 DPI standard scans.
# Absolute floor for low-res captures: 100.0 (ref [9]).
# We render PDFs at 150 DPI (PDF_RENDER_DPI below) — half of 300 DPI standard —
# so we use the absolute floor of 100.0 as our operational minimum.
# Upper bound 900.0 rejects artificially over-sharpened or JPEG-compressed images (ref [10]).
# Pech-Pacheco et al. (ICPR 2000) established the Variance of Laplacian method.
BLUR_THRESHOLD        = 100.0       # Min Laplacian variance. Floor for 150 DPI captures [9].
BLUR_THRESHOLD_UPPER  = 900.0       # Max Laplacian variance. Above = over-sharpened / compressed [10].
BRIGHTNESS_LOW        = 45.0        # Mean pixel value. Below = too dark.
PDF_RENDER_DPI        = 150         # DPI for rendering PDF pages to images.
MAX_FILE_SIZE_BYTES   = 2_097_152   # 2MB hard limit. Reject before any processing.

# Glare detection — ISO/IEC 29794-5 local contrast collapse method
# Research (Rodin & Orlov 2019 [11], ISO/IEC 29794-5 [19]) confirms that static pixel
# thresholds are "fundamentally flawed for mobile capture" because smartphone ISPs map
# clean white paper to 240-255 brightness, causing massive false positives.
# Correct method per ISO/IEC 29794-5: flag a zone as glare when its mean is high AND
# its local std_deviation drops >80% compared to the document baseline (adjacent zones).
# This catches real glare (ink merges with blown-out background, contrast collapses)
# while ignoring clean white paper (high mean but consistent with surrounding areas).
GLARE_STD_DROP_RATIO  = 0.80        # Std deviation must drop >80% vs baseline to flag as glare [19].
GLARE_MEAN_MIN        = 230.0       # Zone mean must exceed this to be considered bright enough for glare.
GLARE_CONTENT_MIN     = 0.10        # Zone must have >10% dark pixels (content) to be evaluated.

# Document fill ratio thresholds
# ICAO Doc 9303 Part 3 [20] mandates 300 DPI minimum for MRZ OCR-B font.
# At 300 DPI on 1920x1080, fill ratio minimum is 25-30% (mathematically derived).
# However: we process scanned PDFs rendered at 150 DPI (half resolution).
# A PAN card (85.6x54mm) on A4 at 150 DPI occupies ~7.5% of the image area.
# Therefore our fill thresholds are correctly calibrated for scanned PDF inputs,
# not for phone-camera capture where 25-30% would apply.
FILL_RATIO_MIN        = 0.02        # Hard fail: document is a tiny speck (< 2%). Calibrated for PDF renders.
FILL_RATIO_WARN       = 0.04        # Soft warning: document small in frame (< 4%). May still be readable.

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

DECISION RULES — apply in strict order, stop at first match:

RULE 1 — 4th CHARACTER IS THE HARD RULE (overrides all visual signals):
  If PAN number is readable AND the 4th character is P:
    → INDIVIDUAL PAN — accept immediately.
    → Do NOT require Signal 3 (visible father's name label).
    → Older PAN card formats (pre-2017) do not show English field labels
       for Name or Father's Name — this is a known format variant, NOT
       evidence of a company card. Photo + 4th char P = individual.

RULE 2 — NON-INDIVIDUAL 4th CHARACTER:
  If PAN number is readable AND 4th character is F, C, H, A, B, G, J, L, or T:
    → NON-INDIVIDUAL PAN — reject with the specific entity type detected.

RULE 3 — PAN NUMBER NOT READABLE BUT PHOTO PRESENT:
  If PAN number cannot be clearly read but a cardholder photo is visible:
    → INDIVIDUAL PAN — accept. Photos never appear on company/entity PAN cards.

RULE 4 — NOTHING READABLE:
  → Reject — cannot confirm individual PAN. Request clearer image.

Output ONLY this JSON with no other text — always include pan_number_extracted:
{"match": true, "detected_type": "pan", "pan_number_extracted": "AWAPJ9107H", "reason": "confirmed — individual PAN, [signals found]"}
or
{"match": false, "detected_type": "pan", "pan_number_extracted": "AWAPJ9107H or null if not readable", "reason": "Company/Firm PAN detected (entity PAN, not individual PAN) — this slot requires the individual cardholder's personal PAN card"}
or
{"match": false, "detected_type": "one of: passport|visa|reload|transfer|ticket|annexure|sof|unknown", "pan_number_extracted": null, "reason": "wrong document — [what this actually appears to be]"}"""
    },
    "visa": {
        "name": "Visa (e-Visa or sticker visa)",
        "signals": [
            "Visa sticker (VISADO/VISA, VISA/VISUM) affixed to passport page, with validity dates and visa number",
            "e-Visa printout with 'Visa issue date' / 'Valid till' fields and applicant details",
            "MRZ at bottom of sticker starting with VC (Schengen), VA (transit), or VR (residence)",
        ],
        "not_this": "Passport bio-data page WITHOUT a visa sticker on it, flight ticket, boarding pass, hotel booking",
        "prompt_override": """You are classifying a document image to verify it is a visa.

CRITICAL — VALID SUBMISSION FORMATS. Both of these are correct:

FORMAT 1 — e-Visa printout:
  A printed PDF page with fields like 'Visa issue date', 'Valid till', 'Type of Visa',
  applicant name, photo, and issuing authority branding.

FORMAT 2 — Visa sticker on passport page (MOST COMMON IN INDIA):
  Applicants photograph their passport opened to the page where the visa sticker is affixed.
  This looks like a passport page WITH a coloured/holographic sticker attached to it.
  DO NOT classify this as 'passport' — the STICKER is the visa.
  Recognise it by:
    - A rectangular sticker with 'VISADO/VISA', 'VISA/VISUM', or country name at top
    - Validity dates on the sticker (FROM / TO or DE / AL or VON / BIS)
    - A visa number (typically 9 digits)
    - Two lines of MRZ text at the bottom of the sticker starting with VC, VA, VR, or V<

FORMAT 3 — Visa grant letter / grant notification:
  Used by Australia, New Zealand, UK (BRP), and others that do not issue sticker visas.
  Shows: Visa Grant Number, applicant name and passport number, visa class/subclass,
  visa conditions, travel validity period, and government immigration branding.
  Key identifiers: "Visa Grant Number", "Subclass", "Department of Home Affairs",
  "Immigration New Zealand", "Grant Date", or "travel to [country] until".
  This IS a valid visa — do NOT classify as wrong document.


SIGNAL 1 — Visa sticker header on any page:
  Look for 'VISADO/VISA', 'VISA/VISUM', 'VISTO', 'SCHENGEN VISA', or
  country flag + country name in the context of a visa sticker rectangle.

SIGNAL 2 — Visa MRZ (sticker visas only):
  Two lines of OCR-B text at the bottom of the sticker.
  Line 1 starts with V followed by document subtype code and country (e.g. VCESP, VCCHE, VKGBR).
  This is DIFFERENT from passport MRZ which starts with P<IND.

SIGNAL 3 — Validity dates visible anywhere on document:
  Any combination of FROM/TO dates, issue date + expiry date, or
  VALIDO DESDE / HASTA, VON / BIS, with a visa number nearby.

DECISION RULES:
  - Visa sticker visible on a passport page → ACCEPT as visa (do NOT call this a passport)
  - e-Visa PDF with validity dates → ACCEPT as visa
  - Visa grant letter / grant notification (Visa Grant Number + validity + issuing authority) → ACCEPT as visa
  - Passport bio-data page with no sticker attached → REJECT (this is passport, not visa)
  - Plain passport page with entry/exit stamps only (no sticker) → REJECT
  - Flight ticket, hotel booking, boarding pass → REJECT

Output ONLY this JSON:
{\"match\": true, \"detected_type\": \"visa\", \"reason\": \"confirmed — [format: sticker on passport/e-visa, signals found]\"}
or
{\"match\": false, \"detected_type\": \"one of: passport|pan|reload|transfer|ticket|annexure|sof|unknown\", \"reason\": \"wrong document — [what this is]\"}"""
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
            "UTR number or transaction reference number confirming a completed fund transfer",
            "INR transfer amount and beneficiary / payee name visible",
            "Bank name or logo present with transaction date",
        ],
        "not_this": "Multi-row bank statement with many transactions, UPI screenshot without UTR, cheque, demand draft",
        "prompt_override": """You are classifying a document to verify it is a bank transfer confirmation.

VALID FORMATS — all of these confirm a bank transfer was completed:
  - HDFC/ICICI/Axis netbanking: 3-step UI with "Processing Successful" and UTR number
  - SBI/PNB NEFT/IMPS: Simple acknowledgement page with transaction reference number
  - Bank advice receipt (e.g. "ICICI Bank Advice Receipt"): single transaction details
  - Payment summary or transfer summary: shows amount, beneficiary, reference, date
  - UPI transfer confirmation: shows UPI transaction ID, amount, payee name
  - Any document that shows: amount transferred + reference/UTR number + beneficiary

INVALID FORMATS:
  - Multi-row bank account statement (has many transactions, opening/closing balance) → this is SOF
  - Cheque image → not a transfer confirmation
  - Demand draft → not an online transfer
  - Salary slip → not a transfer

THE KEY IDENTIFIERS for a valid bank transfer confirmation:
  1. A single transaction reference number (UTR: 12-22 alphanumeric, or transaction ID)
  2. Amount in INR
  3. Payee / beneficiary name (should be Thomas Cook India Ltd or similar)
  4. Transaction date
  5. Bank identity (logo, letterhead, or portal branding)
  ALL FIVE do not need to be present — any 3 of the 5 is sufficient.

DECISION RULE:
  - Contains transaction reference + amount + at least one other identifier → ACCEPT
  - Multi-row statement with 10+ transactions → REJECT as "sof"
  - None of the identifiers present → REJECT as "unknown"

Output ONLY this JSON:
{"match": true, "detected_type": "transfer", "reason": "confirmed — [which identifiers found]"}
or
{"match": false, "detected_type": "one of: sof|passport|pan|visa|reload|ticket|annexure|unknown", "reason": "wrong document — [what this is]"}"""
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
        "prompt_override": """You are classifying a document to verify it belongs in the Annexure slot.

TWO DOCUMENT TYPES ARE VALID OR NEAR-VALID for this slot:

TYPE A — ANNEXURE / DEALER DECLARATION LETTER (exact match):
  A letter from an RBI-authorised forex dealer declaring they are reloading a
  Thomas Cook travel card for a customer. Key identifiers:
  - RBI Authorised Dealer Licence number (format: DEL.FFMC/XXX/YYYY or similar)
  - Phrase confirming the customer 'has to reload' or 'wishes to reload' a Thomas Cook Card
  - Thomas Cook India bank account number (57500000424836) and UTR/reference number
  - Customer details: name, passport number, PAN, travel date, currency, amount

TYPE B — FOREX CARD ISSUANCE LETTER (similar but different — classify separately):
  A letter from a sub-dealer (e.g. Competent Forex Services, United Forex Services)
  to Thomas Cook India Ltd confirming they HAVE ISSUED a forex card.
  Key identifiers:
  - Company letterhead (forex services company name, address, phone)
  - Addressed TO Thomas Cook India Ltd / the authorised dealer
  - Structured table with: Card Holder Name, Card Number, Passport No, PAN,
    Travel Date, Destination, Currency & Amount, UTR Reference
  - FEMA compliance statement at the bottom
  - Authorised Signatory with company stamp

DECISION RULES:
  - TYPE A (Dealer Declaration) → match: true, detected_type: "annexure"
  - TYPE B (Forex Issuance Letter) → match: false, detected_type: "dealer_issuance_letter"
  - Thomas Cook reload form (has 10-currency table, Thomas Cook branding) → match: false, detected_type: "reload"
  - Bank statement, salary slip → match: false, detected_type: "sof"
  - Something else entirely → match: false, detected_type: "unknown"

Output ONLY this JSON — no other text:
{"match": true, "detected_type": "annexure", "reason": "confirmed — RBI licence + Thomas Cook reload declaration found"}
or
{"match": false, "detected_type": "dealer_issuance_letter", "reason": "forex card issuance letter — sub-dealer to Thomas Cook, has card details table and FEMA statement"}
or
{"match": false, "detected_type": "one of: reload|sof|passport|pan|visa|transfer|ticket|unknown", "reason": "wrong document — [what this is]"}"""
    },
    "sof": {
        "name": "Source of Funds — Bank Account Statement",
        "signals": [
            "Multi-row transaction table showing multiple credits and debits over a period",
            "Account holder name and account number visible",
            "Balance information visible (running balance, closing balance, or summary)",
        ],
        "not_this": "Single-transaction bank transfer confirmation, salary slip, credit card statement, investment statement",
        "prompt_override": """You are classifying a document to verify it is a bank account statement (Source of Funds).

VALID SOF FORMATS — all of these are acceptable:
  FORMAT 1 — Digital bank statement (private banks: HDFC, ICICI, Axis, Kotak):
    - "STATEMENT SUMMARY" section with Opening Balance, Closing Balance, Dr/Cr Count
    - "This is a computer generated statement" disclaimer
    - Multi-row transaction table

  FORMAT 2 — Passbook or passbook printout (SBI, PNB, Canara, co-operative banks):
    - Physical passbook pages photographed or scanned
    - Or passbook-style printout from branch
    - Shows: Date, Description/Narration, Withdrawal/Dr, Deposit/Cr, Balance columns
    - NO "computer generated" disclaimer (this is normal for passbooks)
    - NO "STATEMENT SUMMARY" section (this is normal for passbooks)
    - A running balance column in the transaction table IS sufficient

  FORMAT 3 — Bank statement PDF without summary section:
    - Multi-row transaction history
    - Account holder name and account number
    - Date range covered by the statement
    - Balance visible somewhere in the document

WHAT MATTERS — a document is valid SOF if it shows:
  1. Multiple transactions (at least 3-5 rows) covering a period of time
  2. Account holder name or account number
  3. Some form of balance information (running balance OR opening/closing OR current balance)
  The account does NOT need a "STATEMENT SUMMARY" header.
  The document does NOT need a "computer generated" disclaimer.

INVALID as SOF:
  - Single transaction bank advice/receipt (only shows one transaction) → this is "transfer"
  - Salary slip or payslip → "unknown"
  - Credit card statement → "unknown"
  - Fixed deposit receipt or RD passbook → "unknown"

DECISION RULE:
  - Multiple transaction rows + account identity + any balance info → ACCEPT
  - Only one transaction visible → REJECT as "transfer"
  - No transaction rows at all → REJECT

Output ONLY this JSON:
{"match": true, "detected_type": "sof", "reason": "confirmed — [format: digital statement/passbook/other, key signals found]"}
or
{"match": false, "detected_type": "one of: transfer|passport|pan|visa|reload|ticket|annexure|unknown", "reason": "wrong document — [what this is]"}"""
    },
}

# Pairs where a mismatch triggers a soft warning (human review) rather than hard reject.
# These are genuinely similar document types where GPT-4o-mini or the user could reasonably
# confuse them, especially on partial scans.
_SOFT_MISMATCH_PAIRS = {
    frozenset({"annexure", "reload"}),              # Both mention Thomas Cook
    frozenset({"transfer", "sof"}),                 # Both are bank documents
    frozenset({"annexure", "dealer_issuance_letter"}),  # R-ANNEXURE-01: issuance letter ≠ wrong doc
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
    Variance of Laplacian sharpness gate.

    Threshold sources:
    - Method: Pech-Pacheco et al. (ICPR 2000) Variance of Laplacian.
    - Lower bound 100.0: production floor for 150 DPI captures (Dynamsoft DIQA [9]).
      At 300 DPI the standard is 200.0; halved proportionally for our 150 DPI renders.
    - Upper bound 900.0: rejects artificially sharpened or JPEG-compressed images [10].
    - Blank page special case: score < 10 + mean > 235 = no edges, uniform white.

    NIST IR-6101 [23] confirmed: OCR accuracy degrades sharply below 150 DPI
    equivalent sharpness, which corresponds to our 100.0 lower bound.
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
            f"(sharpness: {score:.1f}, minimum: {BLUR_THRESHOLD}). "
            f"[Dynamsoft DIQA, NIST IR-6101]"
        ), score
    # No upper ceiling: PDF-to-image renders at 150 DPI produce naturally high
    # Laplacian scores (1000-6000+) from crisp vector text edges. The research
    # ceiling of 900 [Kashyap ref 10] targets JPEG phone photos with compression
    # artefacts — it must not be applied to PDF renders.
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


# ── Glare detection ─────────────────────────────────────────────────────────────

def _check_glare(gray):
    """
    Detect glare via LOCAL CONTRAST COLLAPSE per ISO/IEC 29794-5 and
    Rodin & Orlov (2019) "Fast Glare Detection in Document Images".

    Static pixel thresholds (e.g. pixel > 253) are fundamentally unreliable
    because smartphone ISPs and scanners routinely map clean white paper to
    240-255 brightness, producing massive false positives on non-glare images.

    The ISO/IEC 29794-5 standard (section on photometric quality) defines glare
    as: a localized zone where peak luminance is high AND local standard deviation
    drops >80% compared to adjacent document zones. This reflects the physical
    mechanism -- glare merges ink strokes with the background, collapsing local
    contrast. Clean white paper has consistently high brightness across ALL zones
    and does not exhibit this selective contrast collapse.

    Algorithm:
    1. Divide image into 3x3 grid of zones.
    2. For each zone with sufficient content (>10% dark pixels), compute mean and std.
    3. Derive baseline std from the median of all content zones.
    4. Flag zones where mean > GLARE_MEAN_MIN AND std has collapsed > GLARE_STD_DROP_RATIO
       relative to the baseline.

    Sources: ISO/IEC 29794-5:2025 [19], Rodin & Orlov 2019 arXiv:1911.05189 [11],
             ISO/IEC 19794-5 dynamic range requirement [19].

    Returns: (passed: bool, reason: str or None, max_std_drop: float)
    """
    h, w = gray.shape
    zh, zw = h // 3, w // 3

    # Pass 1: collect stats for all content-bearing zones
    zone_stats = []
    for row in range(3):
        for col in range(3):
            zone = gray[row * zh:(row + 1) * zh, col * zw:(col + 1) * zw]
            if zone.size == 0:
                continue
            content_ratio = float(np.sum(zone < 200)) / zone.size
            if content_ratio < GLARE_CONTENT_MIN:
                continue   # skip empty margins -- white paper, not glare
            zone_stats.append({
                "pos":     (row + 1, col + 1),
                "mean":    float(np.mean(zone)),
                "std":     float(np.std(zone)),
                "content": content_ratio,
            })

    if not zone_stats:
        return True, None, 0.0   # no content zones -- cannot assess glare

    # Baseline = median std across all content zones (robust to outliers)
    baseline_std = float(np.median([z["std"] for z in zone_stats]))

    if baseline_std < 1.0:
        # Entire document has collapsed contrast -- already captured by blur/CG-DIQA
        return True, None, 0.0

    # Pass 2: flag zones with high mean AND collapsed std
    worst_drop = 0.0
    worst_pos  = None
    for z in zone_stats:
        if z["mean"] < GLARE_MEAN_MIN:
            continue   # zone is not bright enough to be glare
        std_drop = (baseline_std - z["std"]) / baseline_std
        if std_drop > worst_drop:
            worst_drop = std_drop
            worst_pos  = z["pos"]

    if worst_drop > GLARE_STD_DROP_RATIO:
        return False, (
            f"Glare detected in zone {worst_pos} of a 3x3 grid — "
            f"local contrast collapsed {worst_drop * 100:.0f}% below document baseline "
            f"(baseline std: {baseline_std:.1f}). "
            f"Retake without flash — avoid reflections on holograms or lamination. "
            f"[ISO/IEC 29794-5, Rodin & Orlov 2019]"
        ), worst_drop

    return True, None, worst_drop


# ── Document fill ratio ──────────────────────────────────────────────────────────

def _check_fill_ratio(gray):
    """
    Measure what fraction of the image contains non-background content.

    A document photographed from too far away will have a low fill ratio --
    text is unreadably small even if the global sharpness score is fine
    (desk edges or background can be sharp while the document is tiny).

    Uses a 240-brightness threshold to count non-white pixels as content.
    Works for both scanned and photographed documents.

    Returns: (passed: bool, reason: str or None, fill_ratio: float)
    """
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    fill_ratio = float(np.sum(binary > 0)) / binary.size

    if fill_ratio < FILL_RATIO_MIN:
        return False, (
            f"Document occupies only {fill_ratio * 100:.1f}% of the image "
            f"— document may be too small to read reliably "
            f"(minimum: {FILL_RATIO_MIN * 100:.0f}%). "
            f"Move the camera closer or re-scan at higher resolution."
        ), fill_ratio

    if fill_ratio < FILL_RATIO_WARN:
        return True, (
            f"Document fill is low ({fill_ratio * 100:.1f}%) -- document may be "
            f"small in the frame. Consider retaking with the document filling "
            f"most of the frame."
        ), fill_ratio

    return True, None, fill_ratio



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

    # Glare: SOFT WARNING only, not a hard fail.
    # Rationale: glare detection on scanned PDFs is inherently unreliable —
    # zones with sparse content (footers, borders, white space) have naturally
    # low local std_dev compared to text-dense zones, triggering false positives.
    # GPT-4o-mini often reads through moderate glare successfully.
    # Blocking on glare creates more false rejections than it prevents bad extractions.
    # Flag for human review instead — the reviewer can judge the physical document.
    _glare_passed, glare_reason, scores["glare"] = _check_glare(gray)
    glare_warning = glare_reason if not _glare_passed else None

    passed, fill_reason, scores["fill_ratio"] = _check_fill_ratio(gray)
    if not passed:
        return {
            "pass": False, "reason": fill_reason,
            "failed_check": "fill_ratio", "scores": scores, "tier": 1
        }
    # Fill ratio soft warning: pass but flag for review
    fill_warning = fill_reason if (passed and fill_reason) else None
    soft_warning = glare_warning or fill_warning

    # ── Tier 1B: CG-DIQA ───────────────────────────────────────────────────────
    cgdiqa_score, n_regions, status = _check_cgdiqa(gray)
    scores["cgdiqa"]       = round(cgdiqa_score, 2)
    scores["mser_regions"] = n_regions

    if status == "pass":
        return {
            "pass": True,
            "flagged_for_review": bool(soft_warning),
            "reason": soft_warning or "Passes all quality checks",
            "failed_check": None,
            "scores": scores,
            "tier": 1
        }

    if status == "review":
        _borderline_msg = (
            f"Document quality is borderline — security patterns, photograph noise, or "
            f"photocopy quality may reduce OCR accuracy "
            f"(CG-DIQA score: {cgdiqa_score:.1f}, clean threshold: {CGDIQA_REVIEW}). "
            f"Recommend human review before processing."
        )
        return {
            "pass": True,
            "flagged_for_review": True,
            "reason": (soft_warning + " — " if soft_warning else "") + _borderline_msg,
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

        # R-PAN-01 hard rule: 4th character of PAN number is definitive.
        # Override GPT classification if PAN number contradicts it.
        if expected_type == "pan":
            import re as _re
            pan_raw = result.get("pan_number_extracted") or ""
            pan_num = _re.sub(r"[^A-Z0-9]", "", pan_raw.upper())
            if len(pan_num) >= 4:
                fourth = pan_num[3]
                if fourth == "P" and not result.get("match"):
                    result["match"]         = True
                    result["detected_type"] = "pan"
                    result["severity"]      = None
                    result["reason"]        = (
                        f"Individual PAN confirmed by 4th character hard rule: "
                        f"PAN '{pan_num}' position 4 = 'P' (Person). "
                        f"GPT visual classification overridden."
                    )
                elif fourth in "FCHABLGJT" and result.get("match"):
                    result["match"]         = False
                    result["detected_type"] = "pan"
                    result["severity"]      = "hard"
                    result["reason"]        = (
                        f"Non-individual PAN by 4th character rule: "
                        f"PAN '{pan_num}' position 4 = '{fourth}'."
                    )

        return result

    except Exception as e:
        return {
            "match": True,
            "detected_type": None,
            "severity": None,
            "reason": f"Document type check unavailable ({type(e).__name__}) — skipped"
        }


# ── Public API ─────────────────────────────────────────────────────────────────

def _check_visa_quality(images: list):
    """
    Visa-specific extractability check — runs AFTER type check confirms it is a visa.

    Standard DIQA (CGDIQA score) is insufficient for visa images because:
    - Passport page text generates high gradient scores even when the visa
      sticker is tiny, tilted, or occupies a small fraction of the frame.
    - Holographic overlays always obscure VIZ date fields — the MRZ is the
      only reliable source for expiry date. If Tesseract cannot read it,
      the entire visa extraction is unreliable.

    Two independent gates, either causes immediate rejection:

    Gate 1 — Frame fill (< 20% non-white pixels):
      A correctly photographed visa should have the sticker filling most
      of the frame. Low non-white ratio means the passport/sticker is tiny
      in the image, off-angle, or the frame is mostly empty space.
      Empirical baseline: this visa image scored 8.4% — correctly rejected.

    Gate 2 — MRZ detectability (< 2 lines found by Tesseract):
      Sticker visas have a 2-line MRZ outside the holographic zone.
      If Tesseract cannot find both lines, the image is too tilted,
      too small, or too blurry for the Tesseract-MRZ extraction path.
      Without MRZ, expiry date falls back to GPT reading holographic
      VIZ fields — this reliably produces wrong dates.

    Returns None if visa is extractable, or a rejection dict if not.
    """
    if not images:
        return None

    try:
        from PIL import Image as PILImage

        img  = images[0]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── e-Visa detection: skip sticker checks for digital printouts ────
        # e-Visa / entry permit PDFs are text on white paper — naturally low fill.
        # Sticker-specific checks (frame fill, MRZ zone) don't apply.
        # Detect by OCR: if e-Visa keywords are present → pass quality check.
        try:
            _pil_ev  = PILImage.fromarray(gray)
            _ev_text = pytesseract.image_to_string(_pil_ev, config="--psm 11").lower()
            _ev_kw   = [
                "evisa", "e-visa", "entry permit", "electronic travel",
                "visa grant", "grant number", "subclass", "eta ",
                "valid until", "place of birth",    # UAE / AUS e-Visa fields
            ]
            if any(kw in _ev_text for kw in _ev_kw):
                return None   # e-Visa format — no sticker quality checks needed
        except Exception:
            pass   # OCR unavailable — fall through to sticker checks

        # ── Gate 1: Frame fill ────────────────────────────────────────────
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        non_white_ratio = float(np.count_nonzero(thresh)) / (gray.shape[0] * gray.shape[1])

        if non_white_ratio < 0.20:
            return {
                "pass":         False,
                "requires_review": False,
                "reason": (
                    f"Visa document fills only {non_white_ratio:.0%} of the image — "
                    f"too small to extract reliably. "
                    f"Place the passport flat on a table, open to the visa page, and "
                    f"photograph from directly above so the visa sticker fills most of "
                    f"the frame. Minimum recommended fill: 20% of frame area."
                ),
                "failed_check": "visa_frame_fill",
            }

        # ── Gate 2: MRZ detectability ────────────────────────────────────
        _, binary  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pil_img    = PILImage.fromarray(binary)
        raw_text   = pytesseract.image_to_string(
            pil_img,
            config="--psm 11 -c tessedit_char_whitelist="
                   "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
        )
        mrz_lines = [
            ln.strip().replace(" ", "")
            for ln in raw_text.splitlines()
            if len(ln.strip().replace(" ", "")) >= 20 and "<" in ln
        ]

        if len(mrz_lines) < 2:
            return {
                "pass":         False,
                "requires_review": False,
                "reason": (
                    "Visa sticker detected but the machine-readable zone (MRZ) — the two "
                    "lines of text at the bottom of the sticker — cannot be read at this "
                    "quality. Without a readable MRZ, the expiry date cannot be extracted "
                    "reliably. Re-photograph: place the passport flat on a table, open to "
                    "the visa page, and shoot from directly above. Ensure both MRZ lines "
                    "at the bottom of the sticker are fully in-frame, horizontal, and sharp."
                ),
                "failed_check": "visa_mrz_unreadable",
            }

    except Exception:
        # Tesseract or PIL unavailable — skip and allow through
        # (extraction will still attempt; better than blocking on tool failure)
        pass

    return None   # Both gates passed — visa is extractable



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

    failed   = [r for r in page_results if not r["pass"]]
    flagged  = [r for r in page_results if r.get("flagged_for_review")]
    overall_pass = len(failed) == 0

    # Expose CG-DIQA score for routing decisions in calling code
    # Use first page score as representative (most documents are single-page)
    cgdiqa_score = None
    if page_results:
        cgdiqa_score = page_results[0].get("scores", {}).get("cgdiqa")

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

    # ── Visa-specific extractability check ──────────────────────────────────
    # DIQA scores mislead for visa sticker images: the text-rich passport
    # background generates high CGDIQA scores even when the visa sticker is
    # tiny, tilted, or occupies < 10% of the frame.
    # This check gates on actual MRZ readability — the ground truth for
    # whether the visa can produce a reliable expiry date.
    # Runs AFTER type check passes (confirmed to be a visa).
    # Rejects immediately if either gate fails — no flagging, no review.
    if expected_doc_type == "visa" and overall_pass:
        visa_quality = _check_visa_quality(images)
        if visa_quality is not None:
            return {
                "pass":           False,
                "requires_review": False,
                "reason":         visa_quality["reason"],
                "failed_check":   visa_quality["failed_check"],
                "flagged_pages":  [],
                "failed_pages":   [],
                "page_results":   page_results,
                "doc_type_check": doc_type_check,
            }

    return {
        "pass":           overall_pass,
        "requires_review": bool(flagged) if overall_pass else False,
        "flagged_pages":  [r["page"] for r in flagged],
        "cgdiqa_score":   cgdiqa_score,
        "reason": (
            "All pages pass quality checks"
            if overall_pass and not flagged
            else f"{len(flagged)} page(s) flagged for human review — borderline quality"
            if overall_pass and flagged
            else f"{len(failed)} of {len(page_results)} page(s) failed quality checks"
        ),
        "failed_pages":   [r["page"] for r in failed],
        "page_results":   page_results,
        "doc_type_check": doc_type_check,
    }

# ── Fix 3: Anchor field extraction confidence check ────────────────────────────
# Called AFTER extraction (in diqa_test.py), not during DIQA quality gate.
# Validates that key extracted fields match their known regex pattern.
# Returns a soft WARNING -- never rejects a document outright.

import re as _re

_ANCHOR_PATTERNS = {
    "passport": [
        {
            "label":   "Passport Number",
            "key":     "passport_no",
            "pattern": r"^[A-Z][0-9]{7}$",
            "hint":    "Indian passport number must be 1 uppercase letter followed by exactly 7 digits (e.g. Z4565258)",
        },
    ],
    "pan": [
        {
            "label":   "PAN Number",
            "key":     "pan_number",
            "pattern": r"^[A-Z]{5}[0-9]{4}[A-Z]$",
            "hint":    "PAN must be 5 letters + 4 digits + 1 letter (e.g. ACSPV7218B)",
        },
    ],
    "reload": [
        {
            "label":   "PAN Number on Reload Form",
            "key":     "pan_number",
            "pattern": r"^[A-Z]{5}[0-9]{4}[A-Z]$",
            "hint":    "PAN on reload form must be 5 letters + 4 digits + 1 letter -- handwriting OCR may have introduced errors",
        },
    ],
    "visa": [
        {
            "label":   "Visa Number",
            "key":     "visa_number",
            "pattern": r"^[A-Z0-9]{6,15}$",
            "hint":    "Visa number must be 6-15 alphanumeric characters",
        },
    ],
}


def check_extraction_confidence(doc_type: str, extracted: dict) -> list:
    """
    Validate anchor fields in extracted data against known regex patterns.
    Returns a list of warning dicts (empty list = all anchor fields look correct).
    Each warning: {"field": str, "value": str, "hint": str}

    This is a SOFT check -- it flags suspicious values for human review
    but never blocks processing. Designed to catch GPT misreads like:
      - Passport number with wrong character count (Z456512358 instead of Z4565258)
      - PAN number with wrong format (ACS1V7218B instead of ACSPV7218B)
    """
    warnings = []
    patterns = _ANCHOR_PATTERNS.get(doc_type, [])

    for spec in patterns:
        raw = extracted.get(spec["key"])
        if not raw:
            continue  # field absent -- not an anchor check concern

        value = _re.sub(r"[^A-Z0-9]", "", str(raw).upper())
        if not _re.match(spec["pattern"], value):
            warnings.append({
                "field": spec["label"],
                "value": raw,
                "hint":  spec["hint"],
            })

    return warnings