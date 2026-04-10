"""
extract.py  — v4 PDF-PRIMARY
-----------------------------
Unified PDF / plain-text extractor for HealthAI lab reports.

Supports two report modes:
  • Basic      — 6 core fields (glucose, cholesterol, blood_pressure,
                 bmi, bilirubin, age)
  • Full clinical — all of the above PLUS ast, alt, alp,
                 max_heart_rate, oldpeak, ca, fbs, sex, smoking,
                 family_history, chest_pain_type, ecg_result, exang

v4 changes vs v3:
  1. PDF is now the exclusive primary input path — extract_report()
     is the canonical entry point for all field ingestion.
  2. Multi-page aware: tables and raw text are gathered from every
     page before any regex pass, so multi-page lab reports are fully
     supported.
  3. Better table heuristics: rows where col-0 is a numeric value
     (i.e. the label/value columns are swapped) are detected and
     handled.
  4. Unit-stripping: values like "162 mg/dL" or "29 years" are
     cleaned before casting so numbers are never dropped.
  5. extract_pdf_fields() added as a thin public wrapper that returns
     (fields_dict, report_mode) in one call — lets app.py call a
     single function.
  6. get_missing_core_fields() helper added for the /extract route.

Public API
----------
    extract_report(source)            -> dict
    extract_from_text(text)           -> dict
    extract_pdf_fields(source)        -> (dict, str)   # NEW
    detect_report_mode(fields)        -> "basic" | "clinical"
    get_missing_core_fields(fields)   -> list[str]     # NEW
"""

import re
import io
import pdfplumber


# ── Field patterns ────────────────────────────────────────────────────────────
# Each entry: key -> (compiled regex, value_type)
# value_type: "float" | "int" | "str"

_NUM_PATTERNS: dict[str, tuple[re.Pattern, str]] = {
    # ── Core / basic fields ───────────────────────────────────────────────────
    "age": (
        re.compile(r"age[:\s]+(\d+)", re.I), "int"),
    "glucose": (
        re.compile(r"glucose[:\s]+(\d+\.?\d*)", re.I), "float"),
    "cholesterol": (
        re.compile(r"cholesterol[:\s]+(\d+\.?\d*)", re.I), "float"),
    "blood_pressure": (
        re.compile(
            r"(?:diastolic(?:\s*bp)?|blood\s*pressure(?:\s*\(?diastolic\)?)?|"
            r"\bbp\b)[:\s]+(\d+\.?\d*)", re.I), "float"),
    "bmi": (
        re.compile(r"\bbmi[:\s]+(\d+\.?\d*)", re.I), "float"),
    "bilirubin": (
        re.compile(
            r"(?:total\s+)?bilirubin[:\s]+(\d+\.?\d*)", re.I), "float"),

    # ── Extended / clinical fields ────────────────────────────────────────────
    "ast": (
        re.compile(
            r"(?:\bast\b|aspartate(?:\s+amino(?:transferase)?)?)[:\s]+(\d+\.?\d*)",
            re.I), "float"),
    "alt": (
        re.compile(
            r"(?:\balt\b|alanine(?:\s+amino(?:transferase)?)?|sgpt)[:\s]+(\d+\.?\d*)",
            re.I), "float"),
    "alp": (
        re.compile(
            r"(?:\balp\b|alkaline\s*phosph(?:atase)?)[:\s]+(\d+\.?\d*)",
            re.I), "float"),
    "max_heart_rate": (
        re.compile(
            r"(?:max(?:imum)?\s+heart\s+rate|thalch|max\s+hr)[:\s]+(\d+\.?\d*)",
            re.I), "float"),
    "oldpeak": (
        re.compile(r"oldpeak[:\s]+(\d+\.?\d*)", re.I), "float"),
    "ca": (
        re.compile(
            r"(?:ca\b|number\s+of\s+(?:major\s+)?vessels)[:\s]+(\d+\.?\d*)",
            re.I), "float"),
    "fbs": (
        re.compile(
            r"(?:fbs|fasting\s+blood\s+sugar)[:\s]+(\d+\.?\d*)",
            re.I), "float"),
}

_STR_PATTERNS: dict[str, tuple[re.Pattern, list[str]]] = {
    "sex": (
        re.compile(r"\bsex[:\s]+(male|female|m\b|f\b)", re.I),
        ["male", "female"]),
    "smoking": (
        re.compile(
            r"smoking(?:\s+status)?[:\s]+(yes|no|current|former|never|ex(?:\-smoker)?|"
            r"quit|smoker|non-?smoker)", re.I),
        ["yes", "no", "current", "former", "never", "ex", "quit", "smoker"]),
    "family_history": (
        re.compile(
            r"family\s+(?:history|hx)[:\s]+(yes|no|positive|negative)", re.I),
        ["yes", "no", "positive", "negative"]),
    "chest_pain_type": (
        re.compile(
            r"chest\s+pain(?:\s+type)?[:\s]+"
            r"(typical\s+angina|atypical\s+angina|non-?anginal|asymptomatic)",
            re.I),
        ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]),
    "ecg_result": (
        re.compile(
            r"(?:ecg|resting\s+ecg|restecg)[:\s]+"
            r"(normal|st-?t\s+wave(?:\s+abnormality)?|lv\s+hypertrophy)",
            re.I),
        ["normal", "st-t wave", "lv hypertrophy"]),
    "exang_ui": (
        re.compile(
            r"(?:exang|exercise(?:\s+induced)?\s+angina)[:\s]+(yes|no|true|false|1|0)",
            re.I),
        ["yes", "no", "true", "false"]),
}

# ── Canonical label → key mapping (for table extraction) ─────────────────────
_LABEL_MAP: dict[str, str] = {
    # basic
    "age":                          "age",
    "glucose":                      "glucose",
    "cholesterol":                  "cholesterol",
    "blood pressure":               "blood_pressure",
    "blood pressure (diastolic)":   "blood_pressure",
    "diastolic bp":                 "blood_pressure",
    "bp":                           "blood_pressure",
    "bmi":                          "bmi",
    "body mass index":              "bmi",
    "bilirubin":                    "bilirubin",
    "total bilirubin":              "bilirubin",
    # clinical — numeric
    "ast":                          "ast",
    "aspartate aminotransferase":   "ast",
    "alt":                          "alt",
    "alanine aminotransferase":     "alt",
    "sgpt":                         "alt",
    "alp":                          "alp",
    "alkaline phosphatase":         "alp",
    "alkaline phosphotase":         "alp",
    "max heart rate":               "max_heart_rate",
    "maximum heart rate":           "max_heart_rate",
    "thalch":                       "max_heart_rate",
    "oldpeak":                      "oldpeak",
    "ca":                           "ca",
    "number of vessels":            "ca",
    "fbs":                          "fbs",
    "fasting blood sugar":          "fbs",
    # clinical — string
    "sex":                          "sex",
    "gender":                       "sex",
    "smoking":                      "smoking",
    "smoking status":               "smoking",
    "family history":               "family_history",
    "family hx":                    "family_history",
    "chest pain":                   "chest_pain_type",
    "chest pain type":              "chest_pain_type",
    "ecg":                          "ecg_result",
    "ecg result":                   "ecg_result",
    "resting ecg":                  "ecg_result",
    "exang":                        "exang_ui",
    "exercise angina":              "exang_ui",
    "exercise induced angina":      "exang_ui",
}

_BASIC_FIELDS    = {"age", "glucose", "cholesterol", "blood_pressure", "bmi", "bilirubin"}
_CLINICAL_FIELDS = {"ast", "alt", "alp", "max_heart_rate", "oldpeak", "ca", "fbs",
                    "sex", "smoking", "family_history", "chest_pain_type",
                    "ecg_result", "exang_ui"}

_CORE_FIELDS = list(_BASIC_FIELDS)   # the 6 required fields


# ── Internal helpers ──────────────────────────────────────────────────────────

def _strip_units(raw: str) -> str:
    """
    Remove trailing unit strings from a value cell so '162 mg/dL' → '162'.
    Handles common lab units: mg/dL, mmol/L, U/L, IU/L, kg/m², bpm, years, %.
    """
    return re.sub(
        r"\s*(?:mg/d[lL]|mmol/[lL]|[Uu]/[lL]|IU/[lL]|kg/m[²2]|bpm|years?|%)\s*$",
        "", raw.strip(),
    )


def _cast(value_str: str, vtype: str):
    """Cast a raw string to float or int; return None on failure."""
    try:
        cleaned = _strip_units(value_str)
        v = float(cleaned)
        return int(v) if vtype == "int" else v
    except (ValueError, AttributeError):
        return None


def _normalise_str(raw: str, accepted: list[str]) -> str:
    """Lowercase + strip the raw value; map short forms to canonical names."""
    raw = raw.strip().lower()
    if raw in ("m",):              raw = "male"
    if raw in ("f",):              raw = "female"
    if raw in ("ex", "ex-smoker"): raw = "former"
    return raw


def _maybe_swap(label: str, value: str) -> tuple[str, str]:
    """
    Some PDF tables emit rows with value in col-0 and label in col-1.
    Detect that pattern and swap back so extraction logic stays uniform.
    """
    # If the label cell looks numeric and the value cell looks textual, swap.
    try:
        float(_strip_units(label))
        # label is a number — check if value is a known label key
        if value.strip().lower() in _LABEL_MAP:
            return value, label
    except (ValueError, AttributeError):
        pass
    return label, value


def _extract_from_tables(tables: list[list[list]]) -> dict:
    """
    Walk every table row. If col-0 matches a known label, grab col-1 as value.
    Handles tables that have a third column (unit) without breaking.
    Also handles swapped label/value columns.
    """
    results: dict = {}
    for table in tables:
        for row in table:
            if not row or len(row) < 2:
                continue
            raw_label = str(row[0] or "").strip()
            raw_value = str(row[1] or "").strip()
            if not raw_label or not raw_value or raw_value in ("-", "—", "n/a", ""):
                continue

            # Swap detection
            raw_label, raw_value = _maybe_swap(raw_label, raw_value)

            norm_label = raw_label.lower()
            key = _LABEL_MAP.get(norm_label)

            if key is None:
                # Fuzzy: check if any canonical label is a substring
                for canon, mapped_key in _LABEL_MAP.items():
                    if canon in norm_label:
                        key = mapped_key
                        break
            if key is None:
                continue

            str_keys = {"sex", "smoking", "family_history",
                        "chest_pain_type", "ecg_result", "exang_ui"}
            if key in str_keys:
                pattern, accepted = _STR_PATTERNS[key]
                results.setdefault(key, _normalise_str(raw_value, accepted))
            elif key in _NUM_PATTERNS:
                _, vtype = _NUM_PATTERNS[key]
                casted = _cast(raw_value, vtype)
                if casted is not None:
                    results.setdefault(key, casted)

    return results


def _extract_from_raw_text(text: str) -> dict:
    """
    Regex scan over the full text for any field not yet found.
    Numeric patterns first, then string patterns.
    """
    results: dict = {}
    lower = text.lower()

    for key, (pattern, vtype) in _NUM_PATTERNS.items():
        m = pattern.search(lower)
        if m:
            casted = _cast(m.group(1), vtype)
            if casted is not None:
                results[key] = casted

    for key, (pattern, accepted) in _STR_PATTERNS.items():
        m = pattern.search(lower)
        if m:
            results[key] = _normalise_str(m.group(1), accepted)

    return results


# ── Public API ────────────────────────────────────────────────────────────────

def extract_from_text(text: str) -> dict:
    """
    Extract all health fields from a plain text string.
    Suitable for typed-in free text or text already pulled from a PDF.

    Returns
    -------
    dict  {field_name: value}  — numeric fields are float/int,
                                 categorical fields are lowercase strings.
    """
    return _extract_from_raw_text(text)


def extract_report(source) -> dict:
    """
    Extract health values from a PDF file.

    Parameters
    ----------
    source : str | bytes | file-like
        A file path, raw bytes, or a Flask/werkzeug file-object
        (request.files[...]).

    Returns
    -------
    dict  {field_name: value}

    Strategy
    --------
    Multi-page aware:
      1. ALL pages are visited — tables and raw text are gathered first.
      2. _extract_from_tables() runs on the combined table list (higher
         precision — label/value column structure).
      3. _extract_from_raw_text() fills any gaps (higher recall — regex
         over concatenated page text).
    Table-based result wins on key conflicts (setdefault semantics).
    """
    raw_text   = ""
    all_tables = []

    # Accept bytes directly (e.g. from request.get_data())
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)

    with pdfplumber.open(source) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            raw_text += page_text + "\n"
            for table in (page.extract_tables() or []):
                all_tables.append(table)

    # Table extraction takes priority
    results = _extract_from_tables(all_tables)

    # Regex fills any remaining gaps
    regex_results = _extract_from_raw_text(raw_text)
    for k, v in regex_results.items():
        results.setdefault(k, v)

    return results


def extract_pdf_fields(source) -> tuple[dict, str]:
    """
    Convenience wrapper: extract + detect mode in one call.

    Parameters
    ----------
    source : same as extract_report()

    Returns
    -------
    (fields_dict, report_mode)
        fields_dict  : {field_name: value}
        report_mode  : "basic" | "clinical"
    """
    fields = extract_report(source)
    mode   = detect_report_mode(fields)
    return fields, mode


def detect_report_mode(fields: dict) -> str:
    """
    Given the extracted fields dict, return "clinical" if any extended
    clinical field is present, otherwise "basic".
    """
    if any(k in fields for k in _CLINICAL_FIELDS):
        return "clinical"
    return "basic"


def get_missing_core_fields(fields: dict) -> list[str]:
    """
    Return the list of the 6 required core fields that are absent from
    the extracted fields dict. An empty list means the PDF is complete.
    """
    return [f for f in _CORE_FIELDS if f not in fields]