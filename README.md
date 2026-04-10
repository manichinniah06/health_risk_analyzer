# 🫀 HealthAI — Multi-Parameter Health Risk Analyzer

> AI-powered clinical risk assessment using a 3-model Random Forest ensemble · Hackathon-ready

---

## 🏗️ Architecture

```
User Input (6 parameters)
         ↓
┌─────────────────────────────────────────┐
│  🩸 Diabetes Model  → Diabetes Risk %  │  (Glucose, BMI, Age)
│  ❤️  Heart Model    → Heart Risk %     │  (Cholesterol, BP, Age)
│  🫀 Liver Model    → Liver Risk %     │  (Bilirubin, Age)
└─────────────────────────────────────────┘
         ↓  Weighted combination (35% · 40% · 25%)
   Final Risk Score + Level (Low / Medium / High)
         ↓
   Explain + Personalised Recommendations
```

---

## 📁 Project Structure

```
healthai/
├── app.py               ← Flask backend (ensemble prediction, recommendations)
├── train_models.py      ← Trains all 3 ML models (run once)
├── requirements.txt     ← Python dependencies
├── data/
│   ├── diabetes.csv     ← Pima Diabetes dataset
│   ├── heart.csv        ← Cleveland Heart Disease dataset
│   └── liver.csv        ← Indian Liver Patient dataset
├── models/              ← Auto-created after training
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   ├── liver_model.pkl
│   ├── diabetes_metrics.pkl
│   ├── heart_metrics.pkl
│   ├── liver_metrics.pkl
│   └── all_metrics.pkl
└── templates/
    └── index.html       ← Full-featured frontend UI
```

---

## ⚙️ Setup & Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the 3 models (run once)

```bash
python train_models.py
```

### 3. Start Flask

```bash
python app.py
```

### 4. Open in browser

```
http://localhost:5000
```

---

## 🧪 Sample Test Values

### Low Risk

| Parameter | Value |
| --------- | ----- | ------- | --- | ----------- | --- | -------------- | --- | --- | ---- | --------- | --- |
| Age       | 34    | Glucose | 88  | Cholesterol | 175 | Blood Pressure | 72  | BMI | 22.1 | Bilirubin | 0.6 |

### High Risk

| Parameter | Value |
| --------- | ----- | ------- | --- | ----------- | --- | -------------- | --- | --- | ---- | --------- | --- |
| Age       | 63    | Glucose | 198 | Cholesterol | 267 | Blood Pressure | 97  | BMI | 34.7 | Bilirubin | 2.1 |

> Use the **"Low / Medium / High"** preset buttons in the UI.

---

## 📊 API

### POST /predict

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"glucose":198,"cholesterol":267,"blood_pressure":97,"bmi":34.7,"bilirubin":2.1}'
```

### Response

```json
{
  "risk_score": 74.2,
  "risk_level": "High",
  "sub_scores": { "diabetes": 68.1, "heart": 81.3, "liver": 62.5 },
  "contributing_factors": [...],
  "recommendations": [...],
  "lifestyle": [...]
}
```

### GET /metrics

Returns accuracy, precision, recall, F1 for each of the 3 models.

---

## ⚠️ Disclaimer

Educational and demonstration purposes only. Not a substitute for professional medical advice.
