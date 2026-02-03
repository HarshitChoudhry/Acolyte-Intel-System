"""
Acolyte Predictive Student Intelligence - FastAPI Backend

This API provides endpoints for predicting:
1. Visa approval likelihood
2. Student relocation readiness

The prediction uses a two-step stacked model approach.
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import numpy as np
import os
from typing import Optional, List
from src.bulk_handler import process_bulk_csv

# Initialize FastAPI app
app = FastAPI(
    title="Acolyte Predictive Student Intelligence API",
    description="Two-step predictive system for visa approval and relocation readiness",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
visa_scorer = None
relocation_classifier = None


class StudentData(BaseModel):
    """Input schema for student data"""
    gpa: float = Field(..., ge=0.0, le=10.0, description="GPA (0.0 to 10.0 scale)")
    test_score: float = Field(..., ge=0.0, le=100.0, description="Test score (0 to 100)")
    study_gap: int = Field(..., ge=0, description="Study gap in years")
    loan_sanctioned: int = Field(..., ge=0, le=1, description="Loan sanctioned (0 or 1)")
    pof_verified: int = Field(..., ge=0, le=1, description="Proof of funds verified (0 or 1)")
    uni_tier: int = Field(..., ge=1, le=3, description="University tier (1, 2, or 3)")
    visa_refusal: int = Field(..., ge=0, le=1, description="Previous visa refusal (0 or 1)")
    cas_issued: int = Field(..., ge=0, le=1, description="CAS issued (0 or 1)")
    session_time_min: float = Field(..., ge=0, description="Session time in minutes")
    days_to_intake: int = Field(..., ge=0, description="Days until intake")

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Output schema for predictions"""
    visa_likelihood: float = Field(..., description="Visa approval likelihood score (0-100)")
    relocation_readiness_score: float = Field(..., description="Relocation readiness probability (0-1)")
    relocation_readiness_class: str = Field(..., description="Classification: 'High' or 'Low'")
    recommendation: str = Field(..., description="Actionable recommendation")


def load_models():
    """Load both trained models at startup"""
    global visa_scorer, relocation_classifier
    
    try:
        # Determine models directory
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        # Load Visa Scorer
        visa_path = os.path.join(models_dir, 'visa_scorer.pkl')
        with open(visa_path, 'rb') as f:
            visa_scorer = pickle.load(f)
        print(f"✓ Visa Scorer loaded from: {visa_path}")
        
        # Load Relocation Classifier
        relocation_path = os.path.join(models_dir, 'relocation_classifier.pkl')
        with open(relocation_path, 'rb') as f:
            relocation_classifier = pickle.load(f)
        print(f"✓ Relocation Classifier loaded from: {relocation_path}")
        
        print("✅ All models loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load models when the application starts"""
    print("\n" + "=" * 60)
    print("ACOLYTE PREDICTIVE STUDENT INTELLIGENCE API")
    print("=" * 60)
    load_models()
    print("=" * 60 + "\n")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Acolyte Predictive Student Intelligence API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Get visa likelihood and relocation readiness predictions",
            "/health": "GET - Health check endpoint",
            "/docs": "GET - Interactive API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = visa_scorer is not None and relocation_classifier is not None
    
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "visa_scorer": visa_scorer is not None,
        "relocation_classifier": relocation_classifier is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(student: StudentData):
    """
    Predict visa likelihood and relocation readiness for a student.
    
    This endpoint uses a two-step approach:
    1. Step 1: Predict visa approval score using academic and financial features
    2. Step 2: Predict relocation readiness using visa score + engagement metrics
    """
    
    # Check if models are loaded
    if visa_scorer is None or relocation_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please check server logs."
        )
    
    try:
        # Step 1: Prepare features for Visa Scorer
        visa_features = np.array([[
            student.gpa,
            student.test_score,
            student.study_gap,
            student.loan_sanctioned,
            student.pof_verified,
            student.uni_tier,
            student.visa_refusal
        ]])
        
        # Predict visa approval score
        visa_approval_score = visa_scorer.predict(visa_features)[0]
        
        # Normalize to 0-100 scale for user-friendly output
        visa_likelihood = min(max(visa_approval_score, 0), 100)
        
        # Step 2: Prepare features for Relocation Classifier
        # IMPORTANT: Must match training features including weighted_visa_score
        weighted_visa_score = visa_approval_score * 3  # Same 3x weight as training
        
        relocation_features = np.array([[
            visa_approval_score,      # Original visa score
            weighted_visa_score,      # 3x weighted visa score
            student.cas_issued,
            student.session_time_min,
            student.days_to_intake
        ]])
        
        # Predict relocation readiness probability
        relocation_proba = relocation_classifier.predict_proba(relocation_features)[0]
        relocation_readiness_score = relocation_proba[1]  # Probability of high readiness
        
        # Classify
        relocation_class = "High" if relocation_readiness_score >= 0.5 else "Low"
        
        # Generate recommendation
        recommendation = generate_recommendation(
            visa_likelihood,
            relocation_readiness_score,
            student
        )
        
        return PredictionResponse(
            visa_likelihood=round(visa_likelihood, 2),
            relocation_readiness_score=round(relocation_readiness_score, 4),
            relocation_readiness_class=relocation_class,
            recommendation=recommendation
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict-bulk")
async def predict_bulk(file: UploadFile = File(...)):
    """
    Predict visa likelihood and relocation readiness for multiple students via CSV upload.
    """
    # Check if models are loaded
    if visa_scorer is None or relocation_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please check server logs."
        )
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")
    
    try:
        content = await file.read()
        results = process_bulk_csv(content, visa_scorer, relocation_classifier)
        return {"filename": file.filename, "results": results}
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing CSV: {str(e)}"
        )


def generate_recommendation(visa_likelihood: float, relocation_score: float, student: StudentData) -> str:
    """Generate actionable recommendation based on predictions"""
    
    # High visa likelihood, high relocation readiness
    if visa_likelihood >= 70 and relocation_score >= 0.7:
        return "Excellent candidate! Strong visa approval likelihood and high relocation readiness. Proceed with application."
    
    # High visa likelihood, low relocation readiness
    elif visa_likelihood >= 70 and relocation_score < 0.7:
        return "Good visa prospects but low engagement. Increase counseling sessions and provide more intake information."
    
    # Low visa likelihood, high relocation readiness
    elif visa_likelihood < 70 and relocation_score >= 0.7:
        if student.visa_refusal == 1:
            return "High engagement but previous visa refusal detected. Recommend visa interview preparation and documentation review."
        elif student.study_gap >= 3:
            return "High engagement but significant study gap. Prepare strong justification and career progression plan."
        else:
            return "High engagement but moderate visa prospects. Focus on strengthening academic profile and financial documentation."
    
    # Low visa likelihood, low relocation readiness
    else:
        return "Requires attention. Low visa prospects and engagement. Schedule counseling to address concerns and improve documentation."


@app.get("/model-info")
async def model_info():
    """Get information about the loaded models"""
    if visa_scorer is None or relocation_classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded"
        )
    
    return {
        "visa_scorer": {
            "type": "RandomForestRegressor",
            "features": [
                "gpa", "test_score", "study_gap", "loan_sanctioned",
                "pof_verified", "uni_tier", "visa_refusal"
            ],
            "output": "visa_approval_score (0-100)"
        },
        "relocation_classifier": {
            "type": "RandomForestClassifier",
            "features": [
                "visa_approval_score", "cas_issued",
                "session_time_min", "days_to_intake"
            ],
            "output": "relocation_readiness (High/Low)"
        },
        "architecture": "Two-step stacked model (Step 1 output feeds into Step 2)"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    print("\n" + "=" * 60)
    print("Starting Acolyte Intelligence API Server")
    print("=" * 60)
    print(f"Port: {port}")
    print(f"Docs: http://localhost:{port}/docs")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
