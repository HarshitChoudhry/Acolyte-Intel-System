import pandas as pd
import numpy as np
from typing import List, Dict, Any
import io

def process_bulk_csv(file_content: bytes, visa_scorer, relocation_classifier) -> List[Dict[str, Any]]:
    """
    Process a CSV file for bulk predictions.
    
    Expected CSV columns:
    gpa, test_score, study_gap, loan_sanctioned, pof_verified, uni_tier, 
    visa_refusal, cas_issued, session_time_min, days_to_intake
    """
    # Read CSV
    df = pd.read_csv(io.BytesIO(file_content))
    
    # Required columns
    required_cols = [
        'gpa', 'test_score', 'study_gap', 'loan_sanctioned', 
        'pof_verified', 'uni_tier', 'visa_refusal', 'cas_issued', 
        'session_time_min', 'days_to_intake'
    ]
    
    # Check if all required columns are present
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    results = []
    
    for _, row in df.iterrows():
        try:
            # Step 1: Visa Scorer features
            visa_features = np.array([[
                row['gpa'],
                row['test_score'],
                row['study_gap'],
                row['loan_sanctioned'],
                row['pof_verified'],
                row['uni_tier'],
                row['visa_refusal']
            ]])
            
            visa_approval_score = visa_scorer.predict(visa_features)[0]
            visa_likelihood = min(max(visa_approval_score, 0), 100)
            
            # Step 2: Relocation Classifier features
            # IMPORTANT: Must match training features including weighted_visa_score
            weighted_visa_score = visa_approval_score * 3  # Same 3x weight as training
            
            relocation_features = np.array([[
                visa_approval_score,      # Original visa score
                weighted_visa_score,      # 3x weighted visa score
                row['cas_issued'],
                row['session_time_min'],
                row['days_to_intake']
            ]])
            
            relocation_proba = relocation_classifier.predict_proba(relocation_features)[0]
            relocation_readiness_score = relocation_proba[1]
            relocation_class = "High" if relocation_readiness_score >= 0.5 else "Low"
            
            results.append({
                "student_id": row.get('student_id', f"STU_{_}"),
                "gpa": row['gpa'],
                "test_score": row['test_score'],
                "visa_likelihood": round(float(visa_likelihood), 2),
                "relocation_readiness_score": round(float(relocation_readiness_score), 4),
                "relocation_readiness_class": relocation_class,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "student_id": row.get('student_id', f"STU_{_}"),
                "status": "error",
                "error": str(e)
            })
            
    return results
