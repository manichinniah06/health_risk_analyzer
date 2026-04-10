"""
app.py  — v5 INTEGRATED  (PDF-Primary + AI Chatbot)
-----------------------------------------------------
What's new in v5 vs v4:
  1. /chat  endpoint — streams a conversational health assistant powered by
     Groq (llama-3.3-70b-versatile). The chatbot is context-aware: when the
     user has already run a prediction the result JSON can be injected as
     system context so the bot gives personalised, report-specific answers.
  2. GROQ_API_KEY is loaded from the environment (set it in .env or your shell).
  3. All v4 PDF-primary logic is unchanged.
  4. A single new Flask route: POST /chat  { messages: [...], context?: {...} }
     Returns { reply: "..." } or { error: "..." }.
  5. The /chat endpoint validates the Groq key at startup and prints a warning
     (but does NOT crash) if it is missing — the rest of the app still works.

Env vars required for chatbot:
  GROQ_API_KEY=<your key>

All other startup requirements are identical to v4.
"""

import os
import pickle
import json
import re
import numpy as np
import requests

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

load_dotenv()

# ── Import unified extractor (v4) ─────────────────────────────────────────────
from extract import (
    extract_pdf_fields,
    extract_from_text,
    detect_report_mode,
    get_missing_core_fields,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB upload limit

BASE       = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE, "models")

# ── Groq config ───────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    print(
        "\n[STARTUP WARNING] GROQ_API_KEY is not set. "
        "The /chat endpoint will return errors until a valid key is provided.\n"
    )

# ── Chatbot system prompt ─────────────────────────────────────────────────────
CHAT_SYSTEM_PROMPT = """You are HealthAI Assistant, an expert AI health advisor \
embedded in the HealthAI Multi-Parameter Health Risk Analyzer.

Your role:
- Explain what the user's risk scores mean in plain, empathetic language.
- Answer questions about the clinical parameters (glucose, cholesterol, BMI, \
bilirubin, AST/ALT/ALP, blood pressure, heart-rate metrics, etc.).
- Explain the three underlying ML models (Diabetes / Heart / Liver) and how the \
weighted ensemble works.
- Provide general lifestyle and dietary guidance aligned with the recommendations \
already shown in the report.
- Clarify medical terms without giving a clinical diagnosis or prescribing \
medication.

Rules you MUST follow:
- Always remind the user that HealthAI is for educational purposes only and is \
NOT a substitute for professional medical advice.
- Never diagnose, prescribe, or suggest specific drugs or dosages.
- If the user seems to be in medical distress, strongly urge them to call \
emergency services or visit a doctor immediately.
- Keep answers concise (3–6 sentences unless the user asks for detail).
- Use bullet points sparingly — prefer flowing sentences.
- When a prediction context is provided, refer to the user's actual numbers.

If the user greets you or asks an off-topic question, respond warmly but gently \
redirect to health-related topics."""


# ── Safe model / artefact loader ──────────────────────────────────────────────
def _load(name: str):
    path = os.path.join(MODELS_DIR, name)
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise RuntimeError(
            f"Artefact not found: {path}\n"
            f"  → Run train_models.py first to generate all .pkl files."
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load {name}: {exc}") from exc


# ── Load models & artefacts at startup ───────────────────────────────────────
try:
    DIABETES_MODEL    = _load("diabetes_model.pkl")
    HEART_MODEL       = _load("heart_model.pkl")
    LIVER_MODEL       = _load("liver_model.pkl")
    DIABETES_FEATURES = _load("diabetes_features.pkl")
    HEART_FEATURES    = _load("heart_features.pkl")
    LIVER_FEATURES    = _load("liver_features.pkl")
    ALL_METRICS       = _load("all_metrics.pkl")
    DIABETES_MEDIANS  = _load("diabetes_medians.pkl")
    HEART_MEDIANS     = _load("heart_medians.pkl")
    LIVER_MEDIANS     = _load("liver_medians.pkl")
    FEATURE_SCHEMA    = _load("feature_schema.pkl")
except RuntimeError as _e:
    print(f"\n[STARTUP ERROR] {_e}\n")
    raise SystemExit(1)

# Assert feature lists have not drifted since last training run
for _model_key, _features in [
    ("diabetes", DIABETES_FEATURES),
    ("heart",    HEART_FEATURES),
    ("liver",    LIVER_FEATURES),
]:
    _expected = FEATURE_SCHEMA.get(_model_key, [])
    if _features != _expected:
        print(
            f"[STARTUP WARNING] {_model_key} feature list in memory differs "
            f"from feature_schema.pkl — model may be stale. Re-run train_models.py."
        )

# Dummy column lists for one-hot heart features
HEART_DUMMY_COLS = {
    "cp":      [c for c in HEART_FEATURES if c.startswith("cp_")],
    "restecg": [c for c in HEART_FEATURES if c.startswith("restecg_")],
    "slope":   [c for c in HEART_FEATURES if c.startswith("slope_")],
    "thal":    [c for c in HEART_FEATURES if c.startswith("thal_")],
}

# ── Ensemble weights ──────────────────────────────────────────────────────────
WEIGHTS         = {"diabetes": 0.40, "heart": 0.45, "liver": 0.15}
LIVER_BASE_RATE = 0.714

# ── Human-readable field labels ───────────────────────────────────────────────
FIELD_LABELS = {
    "age":              "Age (years)",
    "glucose":          "Glucose (mg/dL)",
    "cholesterol":      "Cholesterol (mg/dL)",
    "blood_pressure":   "Blood Pressure – Diastolic (mmHg)",
    "bmi":              "BMI (kg/m²)",
    "bilirubin":        "Total Bilirubin (mg/dL)",
    "ast":              "AST – Aspartate Aminotransferase (U/L)",
    "alt":              "ALT – Alanine Aminotransferase (U/L)",
    "alp":              "Alkaline Phosphatase (U/L)",
    "max_heart_rate":   "Max Heart Rate (bpm)",
    "oldpeak":          "ST Depression (Oldpeak)",
    "ca":               "Number of Major Vessels (CA)",
    "fbs":              "Fasting Blood Sugar > 120 (0/1)",
    "sex":              "Sex",
    "smoking":          "Smoking Status",
    "family_history":   "Family History of Heart Disease",
    "chest_pain_type":  "Chest Pain Type",
    "ecg_result":       "Resting ECG Result",
    "exang_ui":         "Exercise-Induced Angina",
}

# ── Input validation ranges ───────────────────────────────────────────────────
VALID_RANGES = {
    "age":            (1,    120),
    "glucose":        (20,   700),
    "cholesterol":    (50,   700),
    "blood_pressure": (30,   160),
    "bmi":            (10,    80),
    "bilirubin":      (0,     30),
    "ast":            (0,   5000),
    "alt":            (0,   5000),
    "alp":            (0,   3000),
    "max_heart_rate": (40,   250),
    "oldpeak":        (0,     10),
    "ca":             (0,      4),
    "fbs":            (0,      1),
}


def validate_ranges(features: dict) -> list[str]:
    errors = []
    for field, (lo, hi) in VALID_RANGES.items():
        if field in features and isinstance(features[field], (int, float)):
            v = features[field]
            if not (lo <= v <= hi):
                label = FIELD_LABELS.get(field, field)
                errors.append(
                    f"{label}: value {v} is outside expected range [{lo}, {hi}]")
    return errors


def risk_level(score: float) -> str:
    if score >= 60: return "High"
    if score >= 35: return "Medium"
    return "Low"


# ── Feature vector builders ───────────────────────────────────────────────────

def build_diabetes_vector(f: dict) -> list:
    row = dict(DIABETES_MEDIANS)
    row["Glucose"]       = f.get("glucose",       row["Glucose"])
    row["BMI"]           = f.get("bmi",            row["BMI"])
    row["Age"]           = f.get("age",            row["Age"])
    row["BloodPressure"] = f.get("blood_pressure", row["BloodPressure"])
    return [row[feat] for feat in DIABETES_FEATURES]


def build_heart_vector(f: dict) -> list:
    row = dict(HEART_MEDIANS)

    row["age"]      = f.get("age",            row.get("age",      50))
    row["chol"]     = f.get("cholesterol",    row.get("chol",    200))
    row["trestbps"] = f.get("blood_pressure", row.get("trestbps", 120))
    if "thalch"  in row: row["thalch"]  = f.get("max_heart_rate", row["thalch"])
    if "oldpeak" in row: row["oldpeak"] = f.get("oldpeak",        row["oldpeak"])
    if "ca"      in row: row["ca"]      = f.get("ca",             row["ca"])

    if "sex_enc" in row:
        sv = f.get("sex", "").lower()
        if sv in ("male", "1"):     row["sex_enc"] = 1
        elif sv in ("female", "0"): row["sex_enc"] = 0
    if "fbs_enc" in row:
        row["fbs_enc"] = f.get("fbs", row["fbs_enc"])
    if "exang_enc" in row:
        ev = str(f.get("exang_ui", "")).lower()
        if ev in ("yes", "true", "1"):    row["exang_enc"] = 1
        elif ev in ("no", "false", "0"):  row["exang_enc"] = 0

    cp_input = f.get("chest_pain_type", "").lower().strip()
    CP_MAP = {
        "typical angina":  "cp_typical angina",
        "atypical angina": "cp_atypical angina",
        "non-anginal":     "cp_non-anginal",
        "asymptomatic":    "cp_asymptomatic",
    }
    for col in HEART_DUMMY_COLS.get("cp", []): row[col] = 0.0
    mapped = CP_MAP.get(cp_input)
    if mapped and mapped in row: row[mapped] = 1.0

    ecg_input = f.get("ecg_result", "").lower().strip()
    ECG_MAP = {
        "normal":          "restecg_normal",
        "st-t wave":       "restecg_st-t wave abnormality",
        "lv hypertrophy":  "restecg_lv hypertrophy",
    }
    for col in HEART_DUMMY_COLS.get("restecg", []): row[col] = 0.0
    mapped = ECG_MAP.get(ecg_input)
    if mapped and mapped in row: row[mapped] = 1.0

    return [row.get(feat, 0.0) for feat in HEART_FEATURES]


def build_liver_vector(f: dict) -> list:
    row = dict(LIVER_MEDIANS)
    row["Age"]             = f.get("age",       row["Age"])
    row["Total_Bilirubin"] = f.get("bilirubin", row["Total_Bilirubin"])
    if "Alkaline_Phosphotase"       in row:
        row["Alkaline_Phosphotase"]       = f.get("alp", row["Alkaline_Phosphotase"])
    if "Alamine_Aminotransferase"   in row:
        row["Alamine_Aminotransferase"]   = f.get("alt", row["Alamine_Aminotransferase"])
    if "Aspartate_Aminotransferase" in row:
        row["Aspartate_Aminotransferase"] = f.get("ast", row["Aspartate_Aminotransferase"])
    if "Gender_enc" in row:
        gv = f.get("sex", "").lower()
        if gv in ("male", "1"):     row["Gender_enc"] = 1
        elif gv in ("female", "0"): row["Gender_enc"] = 0
    return [row[feat] for feat in LIVER_FEATURES]


# ── Rule-based recommendations ────────────────────────────────────────────────
def generate_recommendations(f: dict) -> dict:
    contributing, recs, lifestyle = [], [], []
    g   = f.get("glucose", 0)
    ch  = f.get("cholesterol", 0)
    bp  = f.get("blood_pressure", 0)
    bmi = f.get("bmi", 0)
    bi  = f.get("bilirubin", 0)
    age = f.get("age", 0)
    ast = f.get("ast", 0)
    alt = f.get("alt", 0)
    alp = f.get("alp", 0)
    sex     = f.get("sex", "").lower()
    smoking = f.get("smoking", "").lower()
    family  = f.get("family_history", "").lower()
    cp_type = f.get("chest_pain_type", "").lower()

    if g >= 200:
        contributing.append(("Glucose", g, "Dangerously High"))
        recs.append("Consult an endocrinologist immediately for blood sugar management.")
        recs.append("Eliminate sugary drinks, desserts, and refined carbohydrates.")
        lifestyle.append("Monitor blood glucose at least twice daily.")
    elif g >= 140:
        contributing.append(("Glucose", g, "High"))
        recs.append("Reduce sugar and high-glycaemic food intake.")
        lifestyle.append("30 min of moderate aerobic exercise daily helps regulate blood sugar.")
    elif g >= 100:
        contributing.append(("Glucose", g, "Borderline"))
        recs.append("Limit sugary snacks and sweetened beverages.")

    if ch >= 300:
        contributing.append(("Cholesterol", ch, "Dangerously High"))
        recs.append("Seek medical advice for lipid-lowering therapy.")
        recs.append("Avoid all trans fats, full-fat dairy, and red meat.")
        lifestyle.append("Follow a plant-based or Mediterranean diet strictly.")
    elif ch >= 240:
        contributing.append(("Cholesterol", ch, "High"))
        recs.append("Replace saturated fats with omega-3-rich foods (fish, flax, walnuts).")
        lifestyle.append("Aim for at least 150 min of brisk walking per week.")
    elif ch >= 200:
        contributing.append(("Cholesterol", ch, "Borderline"))
        recs.append("Increase dietary fibre (oats, legumes, vegetables) to lower LDL.")

    if bp >= 110:
        contributing.append(("Blood Pressure (diastolic)", bp, "Stage 2 Hypertension"))
        recs.append("Seek urgent medical evaluation for hypertension management.")
        recs.append("Restrict sodium intake to less than 1,500 mg per day.")
        lifestyle.append("Avoid caffeine and alcohol entirely during treatment.")
    elif bp >= 90:
        contributing.append(("Blood Pressure (diastolic)", bp, "High"))
        recs.append("Reduce salt intake and avoid processed foods.")
        lifestyle.append("Practice deep-breathing or meditation daily.")
    elif bp >= 80:
        contributing.append(("Blood Pressure (diastolic)", bp, "Borderline"))
        recs.append("Monitor blood pressure weekly and reduce stress triggers.")

    if bmi >= 40:
        contributing.append(("BMI", bmi, "Severely Obese"))
        recs.append("Consult a bariatric specialist.")
        lifestyle.append("Work with a dietician on a calorie-deficit meal plan.")
    elif bmi >= 30:
        contributing.append(("BMI", bmi, "Obese"))
        recs.append("Structured weight-loss programme targeting 5–10% reduction.")
        lifestyle.append("Combine strength training with cardio 4–5 days per week.")
    elif bmi >= 25:
        contributing.append(("BMI", bmi, "Overweight"))
        recs.append("Aim to reduce BMI below 25 through a balanced calorie deficit.")
        lifestyle.append("Replace sedentary habits with 10-min activity breaks every hour.")
    elif 0 < bmi < 18.5:
        contributing.append(("BMI", bmi, "Underweight"))
        recs.append("Increase caloric intake with nutrient-dense foods.")

    if bi >= 3.0:
        contributing.append(("Bilirubin", bi, "Severely Elevated"))
        recs.append("Urgent liver function panel and hepatologist consultation required.")
        lifestyle.append("Abstain completely from alcohol and hepatotoxic substances.")
    elif bi >= 1.2:
        contributing.append(("Bilirubin", bi, "Elevated"))
        recs.append("Schedule a liver function test; avoid alcohol and excess paracetamol.")
        lifestyle.append("Stay well hydrated (2–3 litres of water daily).")

    if alt > 0:
        if alt >= 200:
            contributing.append(("ALT", alt, "Severely Elevated"))
            recs.append("Significantly elevated ALT suggests hepatocellular damage — seek urgent hepatologist review.")
        elif alt >= 56:
            contributing.append(("ALT", alt, "Elevated"))
            recs.append("Elevated ALT detected. Avoid alcohol, NSAIDs, and hepatotoxic supplements.")
    if ast > 0:
        if ast >= 200:
            contributing.append(("AST", ast, "Severely Elevated"))
            recs.append("High AST may indicate liver or cardiac muscle stress — correlate with other findings.")
        elif ast >= 40:
            contributing.append(("AST", ast, "Elevated"))
            recs.append("Mildly elevated AST — follow up with a full liver panel including GGT.")
    if alp > 0:
        if alp >= 300:
            contributing.append(("Alkaline Phosphatase", alp, "Severely Elevated"))
            recs.append("Highly elevated ALP — investigate for biliary obstruction or bone disease.")
        elif alp >= 120:
            contributing.append(("Alkaline Phosphatase", alp, "Elevated"))
            recs.append("Elevated ALP — consider ultrasound of the biliary system.")

    if smoking in ("yes", "current", "smoker"):
        contributing.append(("Smoking", "Current", "Active Risk Factor"))
        recs.append("Smoking significantly amplifies cardiovascular and lung cancer risk — cessation is the single most impactful change you can make.")
        lifestyle.append("Explore nicotine replacement therapy or varenicline with your doctor.")
    elif smoking in ("former", "ex", "quit"):
        recs.append("Good progress quitting smoking. Continue to avoid secondhand smoke.")

    if family in ("yes", "positive"):
        contributing.append(("Family History", "Positive", "Elevated Genetic Risk"))
        recs.append("Positive family history of heart disease or diabetes increases your baseline risk — earlier and more frequent screening is advised.")
        lifestyle.append("Discuss genetic screening or preventive therapy with your GP.")

    if cp_type == "typical angina":
        contributing.append(("Chest Pain", "Typical Angina", "Cardiac Concern"))
        recs.append("Typical angina is strongly associated with coronary artery disease — cardiology referral is recommended.")
    elif cp_type == "atypical angina":
        recs.append("Atypical angina reported — stress ECG or cardiac imaging may be warranted.")

    if sex == "female" and age >= 50:
        lifestyle.append("Post-menopausal women face increased cardiovascular risk — discuss HRT and lipid management with your doctor.")
    if sex == "male" and age >= 45:
        lifestyle.append("Men over 45 should have annual cardiovascular risk assessments.")

    if age >= 60:
        recs.append("Annual comprehensive health check-up is strongly recommended.")
    if age >= 50:
        lifestyle.append("Include calcium and vitamin D supplementation after medical advice.")

    lifestyle.append("Maintain consistent sleep of 7–9 hours per night.")
    lifestyle.append("Stay socially active and manage mental health proactively.")

    if not contributing:
        recs.append("All parameters appear within normal ranges — keep up the great work!")
        lifestyle.append("Continue regular exercise and balanced nutrition.")

    return {
        "contributing_factors": contributing,
        "recommendations":      recs,
        "lifestyle":            lifestyle,
    }


# ── Shared: parse a single request into a features dict ──────────────────────
def _parse_features(req) -> dict:
    features: dict = {}
    content_type = req.content_type or ""

    if "application/json" in content_type:
        body = req.get_json(force=True, silent=True) or {}
        for k, v in body.items():
            if v not in ("", None):
                try:    features[k] = float(v)
                except: features[k] = v
    else:
        pdf_file = req.files.get("pdf_file")
        if pdf_file and pdf_file.filename.lower().endswith(".pdf"):
            extracted, _ = extract_pdf_fields(pdf_file)
            features.update(extracted)

        if not features:
            text_input = req.form.get("text_input", "").strip()
            if text_input:
                features.update(extract_from_text(text_input))

        numeric_fields = [
            "age", "glucose", "cholesterol", "blood_pressure", "bmi",
            "bilirubin", "ast", "alt", "alp", "max_heart_rate",
            "oldpeak", "ca", "fbs",
        ]
        for field in numeric_fields:
            val = req.form.get(field, "").strip()
            if val:
                try: features[field] = float(val)
                except: pass

        for field in ["sex", "smoking", "family_history",
                      "chest_pain_type", "ecg_result", "exang_ui"]:
            val = req.form.get(field, "").strip()
            if val:
                features[field] = val.lower()

    return features


# ══════════════════════════════════════════════════════════════════════════════
#  NEW v5: Chatbot helper
# ══════════════════════════════════════════════════════════════════════════════

def _build_context_block(context: dict) -> str:
    """
    Convert the prediction result dict (from /predict) into a concise
    natural-language context block that is prepended to the system prompt.
    """
    if not context:
        return ""

    lines = ["--- User's Latest HealthAI Report ---"]

    rs = context.get("risk_score")
    rl = context.get("risk_level")
    if rs is not None:
        lines.append(f"Overall risk score: {rs}% ({rl} Risk)")

    ss = context.get("sub_scores", {})
    if ss:
        lines.append(
            f"Sub-scores — Diabetes: {ss.get('diabetes')}%, "
            f"Heart: {ss.get('heart')}%, Liver: {ss.get('liver')}%"
        )

    inp = context.get("input_features", {})
    if inp:
        labels = context.get("field_labels", FIELD_LABELS)
        param_parts = []
        for k, v in inp.items():
            label = labels.get(k, k)
            param_parts.append(f"{label} = {v}")
        lines.append("Submitted parameters: " + "; ".join(param_parts))

    cf = context.get("contributing_factors", [])
    if cf:
        flags = [f"{c['param']} ({c['value']}) — {c['status']}" for c in cf]
        lines.append("Flagged factors: " + ", ".join(flags))

    dq = context.get("data_quality", {})
    if dq:
        mode = "Full Clinical" if dq.get("clinical_mode") else "Basic"
        lines.append(f"Report mode: {mode} ({dq.get('total_fields_extracted', '?')} fields extracted)")

    lines.append("--- End of Report ---")
    return "\n".join(lines)


def _call_groq(messages: list, system_suffix: str = "") -> str:
    """
    Call Groq chat completions API and return the assistant reply string.
    Raises ValueError with a user-friendly message on failure.
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not configured. Please add it to your .env file."
        )

    system_content = CHAT_SYSTEM_PROMPT
    if system_suffix:
        system_content = system_content + "\n\n" + system_suffix

    payload = {
        "model":       GROQ_MODEL,
        "messages":    [{"role": "system", "content": system_content}] + messages,
        "temperature": 0.55,
        "max_tokens":  700,
    }

    resp = requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json",
        },
        json=payload,
        timeout=30,
    )

    if not resp.ok:
        try:
            err_detail = resp.json().get("error", {}).get("message", resp.text)
        except Exception:
            err_detail = resp.text
        raise ValueError(f"Groq API error ({resp.status_code}): {err_detail}")

    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# ══════════════════════════════════════════════════════════════════════════════
#  Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/metrics")
def metrics():
    return jsonify(ALL_METRICS)


@app.route("/extract", methods=["POST"])
def extract_only():
    """
    Dry-run extraction endpoint — returns extracted fields without running
    the ML models. Used by the frontend to auto-fill the form after PDF upload.

    Response JSON:
    {
        "extracted_fields":    { field: value, ... },
        "report_mode":         "basic" | "clinical",
        "missing_core_fields": [...],
        "field_labels":        { field: "Human Label (unit)", ... }
    }
    """
    features = _parse_features(request)
    mode     = detect_report_mode(features)
    missing  = get_missing_core_fields(features)
    return jsonify({
        "extracted_fields":    features,
        "report_mode":         mode,
        "missing_core_fields": missing,
        "field_labels":        FIELD_LABELS,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint. See v4 docstring for full details.
    Unchanged from v4 — chatbot is layered on top via /chat.
    """
    features = _parse_features(request)

    missing = get_missing_core_fields(features)
    if missing:
        missing_labels = [FIELD_LABELS.get(f, f) for f in missing]
        return jsonify({
            "error":  f"Missing required fields: {', '.join(missing)}",
            "detail": missing_labels,
            "hint":   "Upload a PDF lab report — both basic and clinical fields "
                      "are extracted automatically.",
        }), 400

    range_errors = validate_ranges(features)
    if range_errors:
        return jsonify({
            "error":  "One or more field values are outside physiological ranges.",
            "detail": range_errors,
        }), 422

    mode = detect_report_mode(features)

    dv = build_diabetes_vector(features)
    hv = build_heart_vector(features)
    lv = build_liver_vector(features)

    d_cls = list(DIABETES_MODEL.classes_)
    h_cls = list(HEART_MODEL.classes_)
    l_cls = list(LIVER_MODEL.classes_)

    diabetes_risk  = float(DIABETES_MODEL.predict_proba([dv])[0][d_cls.index(1)]) * 100
    heart_risk     = float(HEART_MODEL.predict_proba([hv])[0][h_cls.index(1)])    * 100
    liver_risk_raw = float(LIVER_MODEL.predict_proba([lv])[0][l_cls.index(1)])    * 100
    liver_risk     = float(np.clip(
        (liver_risk_raw / (LIVER_BASE_RATE * 100)) * 50, 0, 100))

    final = round(
        WEIGHTS["diabetes"] * diabetes_risk +
        WEIGHTS["heart"]    * heart_risk    +
        WEIGHTS["liver"]    * liver_risk, 1)

    level  = risk_level(final)
    advice = generate_recommendations(features)

    liver_fields_provided = [f for f in
        ["ast", "alt", "alp", "sex", "bilirubin", "age"] if f in features]
    heart_fields_provided = [f for f in
        ["sex", "chest_pain_type", "ecg_result", "max_heart_rate",
         "oldpeak", "ca", "cholesterol", "blood_pressure", "age"] if f in features]

    return jsonify({
        "risk_score": final,
        "risk_level": level,
        "sub_scores": {
            "diabetes": round(diabetes_risk, 1),
            "heart":    round(heart_risk,    1),
            "liver":    round(liver_risk,    1),
        },
        "input_features": {k: v for k, v in features.items()},
        "field_labels":   FIELD_LABELS,
        "contributing_factors": [
            {"param": cf[0], "value": cf[1], "status": cf[2]}
            for cf in advice["contributing_factors"]
        ],
        "recommendations": advice["recommendations"],
        "lifestyle":       advice["lifestyle"],
        "data_quality": {
            "report_mode":           mode,
            "liver_fields_provided": liver_fields_provided,
            "heart_fields_provided": heart_fields_provided,
            "clinical_mode":         mode == "clinical",
            "total_fields_extracted": len(features),
        },
    })


# ── NEW v5: Chatbot endpoint ──────────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    """
    AI chatbot endpoint powered by Groq.

    Request JSON:
    {
        "messages": [
            {"role": "user",      "content": "What does my glucose level mean?"},
            {"role": "assistant", "content": "..."},   // optional prior turns
            {"role": "user",      "content": "..."}    // latest message
        ],
        "context": { ...prediction result from /predict... }   // optional
    }

    Response JSON:
    {
        "reply": "...",
        "model": "llama-3.3-70b-versatile"
    }

    Error response:
    {
        "error": "..."
    }
    """
    body = request.get_json(force=True, silent=True) or {}

    messages = body.get("messages", [])
    context  = body.get("context",  {})

    # Validate messages list
    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "messages array is required and must not be empty."}), 400

    # Sanitise — only pass role + content to Groq
    clean_messages = []
    for m in messages:
        role    = m.get("role", "")
        content = m.get("content", "")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            clean_messages.append({"role": role, "content": content.strip()})

    if not clean_messages:
        return jsonify({"error": "No valid user/assistant messages found."}), 400

    # Build optional context block from prediction result
    context_block = _build_context_block(context)

    try:
        reply = _call_groq(clean_messages, system_suffix=context_block)
        return jsonify({"reply": reply, "model": GROQ_MODEL})
    except ValueError as e:
        return jsonify({"error": str(e)}), 502
    except requests.exceptions.Timeout:
        return jsonify({"error": "Groq API timed out. Please try again."}), 504
    except Exception as e:
        return jsonify({"error": f"Unexpected chatbot error: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)