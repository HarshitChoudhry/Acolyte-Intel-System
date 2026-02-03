# Acolyte Predictive Student Intelligence System

A comprehensive **full-stack** two-step predictive system for assessing student visa approval likelihood and relocation readiness using machine learning.

## 🎯 Overview

This system implements a **stacked two-step machine learning pipeline** with a modern React frontend:

1. **Step 1: Visa Approval Score Predictor** (RandomForestRegressor)
   - Predicts visa approval likelihood (0-100%) based on academic and financial factors
   
2. **Step 2: Relocation Readiness Classifier** (RandomForestClassifier)
   - Uses visa score + engagement metrics to predict student relocation readiness
   - **Enhanced with weighted visa score** for better prediction accuracy
   - The output from Step 1 becomes an input feature for Step 2

## ✨ Key Features

- 🎓 **Enhanced Intent Score Formula** - Balanced weights (Academic 30%, Financial 15%, Risk 20%, Engagement 25%, Urgency 10%)
- 📊 **Weighted Visa Score** - 4x importance in relocation classifier for realistic predictions
- 🌐 **Modern React Frontend** - Built with Vite, responsive design, dark theme
- 📤 **Bulk Upload** - Process multiple students via CSV upload
- 📥 **CSV Template Download** - One-click template download with sample data
- 🚀 **FastAPI Backend** - High-performance async API with auto-generated docs
- 📈 **Comprehensive EDA** - Jupyter notebook with visualizations and insights

## 🏗️ Architecture

```
┌─────────────────┐
│  React Frontend │  (Vite + Modern UI)
│  Port: 5175     │
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

### Prediction Flow

```
Student Data → [Visa Scorer] → Visa Score (+ 3x Weighted) → [Relocation Classifier] → Final Prediction
                    ↓                                              ↓
              Academic/Financial                            Engagement Metrics
              (GPA, Test, Gap,                             (CAS, Session Time,
               Loan, POF, Tier,                             Days to Intake)
               Visa Refusal)
```

## 📁 Project Structure

```
acolyte-intel-system/
├── app.py                      # FastAPI backend
├── README.md                   # This file
├── data/
│   └── student_data.csv       # Training dataset (1000 records)
├── models/
│   ├── visa_scorer.pkl        # Trained Step 1 model
│   └── relocation_classifier.pkl  # Trained Step 2 model (with weighted features)
├── src/
│   ├── data_generation.py     # Synthetic data generator (enhanced intent score)
│   ├── train.py               # Model training pipeline (with weighted visa score)
│   └── bulk_handler.py        # CSV bulk processing
├── frontend/                   # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx            # Main app component
│   │   ├── components/
│   │   │   ├── SinglePredictor.jsx  # Single student prediction
│   │   │   └── BulkPredictor.jsx    # Bulk CSV upload
│   │   └── index.css          # Styling
│   ├── package.json
│   └── vite.config.js
└── notebooks/
    └── EDA.ipynb              # Exploratory data analysis
```

## 🚀 Quick Start

### Prerequisites

- **Backend**: Python 3.8+
- **Frontend**: Node.js 16+ and npm

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd acolyte-intel-system
   ```

2. **Backend Setup**
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install dependencies
   pip install fastapi uvicorn scikit-learn pandas numpy python-multipart
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   ```

### Running the Application

#### Option 1: Run Both (Recommended)

**Terminal 1 - Backend:**
```bash
uvicorn app:app --reload
```
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```
- Frontend: http://localhost:5173 (or 5174, 5175 if port is busy)

#### Option 2: Backend Only (API Testing)

```bash
uvicorn app:app --reload
```
Visit http://localhost:8000/docs for interactive API documentation.

### First-Time Setup

If models don't exist, train them first:

```bash
python src/train.py
```

This will:
- Generate synthetic training data (1000 records)
- Train both models (Visa Scorer + Relocation Classifier)
- Save models to `models/` directory

## 📊 API Endpoints

### `POST /predict`

Predict visa likelihood and relocation readiness for a single student.

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "visa_likelihood": 78.45,
  "relocation_readiness_score": 0.8234,
  "relocation_readiness_class": "High",
  "recommendation": "Excellent candidate! Strong visa approval likelihood and high relocation readiness. Proceed with application."
}
```

### `POST /predict-bulk`

Process multiple students via CSV upload.

**Request:**
- Content-Type: `multipart/form-data`
- File field: `file` (CSV file)

**CSV Format:**
```csv
student_id,gpa,test_score,study_gap,loan_sanctioned,pof_verified,uni_tier,visa_refusal,cas_issued,session_time_min,days_to_intake
STU_001,8.5,85,0,1,1,1,0,1,25,90
STU_002,7.2,75,1,1,1,2,0,1,20,120
```

**Response:**
```json
{
  "filename": "students.csv",
  "results": [
    {
      "student_id": "STU_001",
      "gpa": 8.5,
      "test_score": 85,
      "visa_likelihood": 78.45,
      "relocation_readiness_score": 0.8234,
      "relocation_readiness_class": "High",
      "status": "success"
    }
  ]
}
```

### `GET /health`

Check API health and model status.

### `GET /model-info`

Get detailed information about loaded models and architecture.

## 🎓 Model Features & Enhancements

### Step 1: Visa Scorer Features
- **gpa**: Grade Point Average (0.0 - 10.0 scale only)
- **test_score**: Standardized test score (0 - 100)
- **study_gap**: Years of study gap
- **loan_sanctioned**: Education loan sanctioned (0/1)
- **pof_verified**: Proof of funds verified (0/1)
- **uni_tier**: University tier (1=Elite, 2=Mid-range, 3=Emerging)
- **visa_refusal**: Previous visa refusal history (0/1)

### Step 2: Relocation Classifier Features (Enhanced)
- **visa_approval_score**: Output from Step 1 (1x weight)
- **weighted_visa_score**: visa_approval_score × 3 (3x weight) ⭐ **NEW**
- **cas_issued**: CAS document issued (0/1)
- **session_time_min**: Counseling session time
- **days_to_intake**: Days until program start

### Enhanced Intent Score Formula ⭐

The training data uses a balanced intent score calculation:

```python
intent_score = (
    # Academic (30%)
    (gpa / 10.0) × 15 +
    (test_score / 100) × 15 +
    
    # Financial (15%)
    loan_sanctioned × 7.5 +
    pof_verified × 7.5 +
    
    # Risk (20%)
    (1 - visa_refusal) × 10 +
    (1 - study_gap / 5.0) × 10 +
    
    # Engagement (25%)
    session_time_min × 0.5 +
    cas_issued × 10 +
    
    # Urgency (10%)
    (1 - days_to_intake / 180) × 10
)
```

**Benefits:**
- ✅ Academic merit matters (30% vs 0% before)
- ✅ Financial readiness considered (15% vs 0% before)
- ✅ Risk factors included (20% vs 0% before)
- ✅ Engagement balanced (25% vs 80% before)

## 📈 Model Performance

### Visa Scorer (Step 1)
- **Model**: RandomForestRegressor
- **Features**: 7 (academic + financial + risk)
- **Performance**: RMSE ~1.5-2.0 on test set
- **Top Feature**: GPA (69% importance)

### Relocation Classifier (Step 2)
- **Model**: RandomForestClassifier
- **Features**: 5 (including weighted visa score)
- **Performance**: ~97.5% accuracy on test set
- **Enhancement**: Visa score now has 4x total weight (original + 3x weighted)

**Key Improvement:** Low visa likelihood (<30%) now correctly predicts LOW relocation readiness (was incorrectly HIGH before enhancement).

## 🎨 Frontend Features

### Single Prediction
- Interactive form with real-time validation
- Visual progress indicators
- Color-coded results (green for high, red for low)
- Actionable recommendations

### Bulk Upload
- Drag-and-drop CSV upload
- **One-click template download** ⭐
- Results table with filtering
- Toggle to show only high-readiness students
- Export-ready format

### UI/UX
- Modern dark theme with glassmorphism
- Responsive design (desktop + mobile)
- Smooth animations and transitions
- Professional color scheme

## 🧪 Testing the System

### Using the Frontend

1. **Single Prediction:**
   - Navigate to http://localhost:5173
   - Fill in student details
   - Click "Generate Prediction"
   - View results and recommendations

2. **Bulk Upload:**
   - Click "Bulk Upload" tab
   - Download template CSV
   - Fill with student data
   - Upload and view results

### Using cURL

```bash
curl -X POST "http://localhost:8000/predict" \
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
    "http://localhost:8000/predict",
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

## 🚢 Deployment

### Deploy to Render

1. **Create Web Service** on Render dashboard
2. **Connect GitHub repository**
3. **Configure:**
   - Build Command: `pip install fastapi uvicorn scikit-learn pandas numpy python-multipart`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - Environment: Python 3.11

4. **Deploy Frontend** (separate service):
   - Root Directory: `frontend`
   - Build Command: `npm install && npm run build`
   - Publish Directory: `dist`

### Environment Variables

- `PORT`: Server port (auto-set by Render)

## 📝 Development Workflow

### 1. Data Generation
```bash
python src/data_generation.py
```
Generates 1000 synthetic student records with enhanced intent score formula.

### 2. Model Training
```bash
python src/train.py
```
Trains both models with weighted visa score feature.

### 3. Exploratory Analysis
```bash
jupyter notebook notebooks/EDA.ipynb
```
Visualize correlations, distributions, and model validation.

### 4. Backend Development
```bash
uvicorn app:app --reload
```






