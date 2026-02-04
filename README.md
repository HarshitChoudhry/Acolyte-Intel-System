# Acolyte Predictive Student Intelligence System

A full-stack AI-powered system for assessing student visa approval likelihood and relocation readiness using machine learning.

## � Live Application

**🚀 Production:**
- **Frontend:** https://acolyte-frontend.onrender.com
- **API:** https://acolyte-backend.onrender.com
- **API Docs:** https://acolyte-backend.onrender.com/docs

**💻 Local Development:**
- **Frontend:** http://localhost:5173
- **API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

---

## 🎯 Overview

Two-step machine learning pipeline:

1. **Visa Approval Score Predictor** (RandomForestRegressor)
   - Predicts visa approval likelihood (0-100%) based on academic and financial factors
   
2. **Relocation Readiness Classifier** (RandomForestClassifier)
   - Predicts student relocation readiness using visa score + engagement metrics
   - Enhanced with weighted visa score (4x importance) for accurate predictions

---

## 🏗️ Architecture

```
┌─────────────────┐
│  React Frontend │  (Vite + Modern UI)
│  Port: 5173     │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│  FastAPI Backend│  (Python)
│  Port: 8000     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────────┐
│ Visa   │ │ Relocation   │
│ Scorer │→│ Classifier   │
└────────┘ └──────────────┘
```

---

## 📁 Project Structure

```
acolyte-intel-system/
├── app.py                      # FastAPI backend
├── requirements.txt            # Python dependencies
├── data/
│   └── student_data.csv       # Training dataset (1000 records)
├── models/
│   ├── visa_scorer.pkl        # Trained Step 1 model
│   └── relocation_classifier.pkl  # Trained Step 2 model
├── src/
│   ├── data_generation.py     # Synthetic data generator
│   ├── train.py               # Model training pipeline
│   └── bulk_handler.py        # CSV bulk processing
├── frontend/                   # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── SinglePredictor.jsx
│   │   │   └── BulkPredictor.jsx
│   │   └── index.css
│   └── package.json
└── notebooks/
    └── EDA.ipynb              # Exploratory data analysis
```

---

## 🚀 Quick Start

### Prerequisites

- **Backend:** Python 3.8+
- **Frontend:** Node.js 16+ and npm

### Installation

```bash
# Clone repository
git clone https://github.com/HarshitChoudhry/Acolyte-Intel-System.git
cd acolyte-intel-system

# Backend setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd frontend
npm install
```

### Running Locally

**Terminal 1 - Backend:**
```bash
uvicorn app:app --reload
```
Visit: http://localhost:8000/docs

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```
Visit: http://localhost:5173

### First-Time Setup

Train models (if needed):
```bash
python src/train.py
```

---

## 🎓 Input Features

### Visa Scorer (Step 1)
- **GPA** (0.0 - 10.0)
- **Test Score** (0 - 100)
- **Study Gap** (years)
- **Loan Sanctioned** (Yes/No)
- **POF Verified** (Yes/No)
- **University Tier** (1=Elite, 2=Mid, 3=Emerging)
- **Visa Refusal** (Yes/No)

### Relocation Classifier (Step 2)
- **Visa Approval Score** (from Step 1)
- **CAS Issued** (Yes/No)
- **Session Time** (minutes)
- **Days to Intake**

---

## 📊 Model Performance

| Model | Type | Accuracy/RMSE | Key Features |
|-------|------|---------------|--------------|
| Visa Scorer | RandomForestRegressor | RMSE ~1.5-2.0 | GPA (69% importance) |
| Relocation Classifier | RandomForestClassifier | ~97.5% accuracy | Weighted visa score (80%+ importance) |

---

## 🧪 Testing the System

### Using the Live App

**Single Prediction:**
1. Visit https://acolyte-frontend.onrender.com
2. Fill in student details
3. Click "Generate Prediction"

**Bulk Upload:**
1. Click "Bulk Upload" tab
2. Download template CSV
3. Upload your data
4. View results

### Using API (cURL)

```bash
curl -X POST "https://acolyte-backend.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gpa": 8.5,
    "test_score": 85.0,
    "study_gap": 1,
    "loan_sanctioned": 1,
    "pof_verified": 1,
    "uni_tier": 2,
    "visa_refusal": 0,
    "cas_issued": 1,
    "session_time_min": 25.0,
    "days_to_intake": 90
  }'
```

### Using Python

```python
import requests

response = requests.post(
    "https://acolyte-backend.onrender.com/predict",
    json={
        "gpa": 8.5,
        "test_score": 85.0,
        "study_gap": 1,
        "loan_sanctioned": 1,
        "pof_verified": 1,
        "uni_tier": 2,
        "visa_refusal": 0,
        "cas_issued": 1,
        "session_time_min": 25.0,
        "days_to_intake": 90
    }
)
print(response.json())
```

**Expected Response:**
```json
{
  "visa_likelihood": 78.45,
  "relocation_readiness_score": 0.8234,
  "relocation_readiness_class": "High",
  "recommendation": "Excellent candidate! Strong visa approval likelihood and high relocation readiness."
}
```

---

## � Development Workflow

### 1. Data Generation
```bash
python src/data_generation.py
```

### 2. Model Training
```bash
python src/train.py
```

### 3. Exploratory Analysis
```bash
jupyter notebook notebooks/EDA.ipynb
```

### 4. Backend Development
```bash
uvicorn app:app --reload
```

### 5. Frontend Development
```bash
cd frontend
npm run dev
```

---


## 📊 Data Schema

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| gpa | float | 0.0-10.0 | Grade Point Average |
| test_score | float | 0-100 | Standardized test score |
| study_gap | int | 0-5 | Years of study gap |
| loan_sanctioned | int | 0/1 | Education loan status |
| pof_verified | int | 0/1 | Proof of funds verified |
| uni_tier | int | 1-3 | University tier |
| visa_refusal | int | 0/1 | Previous visa refusal |
| cas_issued | int | 0/1 | CAS document issued |
| session_time_min | float | 0-31 | Counseling session duration |
| days_to_intake | int | 15-180 | Days until program start |

---

## 🎯 Use Cases

### ✅ Ideal For:
- Post-engagement student monitoring
- Hybrid assessment (academic + engagement)
- Risk identification and intervention
- Batch processing of applicants

### ⚠️ Limitations:
- Requires engagement data (CAS, session time)
- Best used after initial counseling sessions

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request


