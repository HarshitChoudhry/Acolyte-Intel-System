import pandas as pd
import numpy as np

def generate_enhanced_data(n=1000):
    np.random.seed(42)
    
    # Existing Features
    gpa = np.random.uniform(5.0, 10.0, n)  # GPA on 10-point scale
    test_score = np.random.normal(75, 10, n).clip(50, 100)
    uni_tier = np.random.choice([1, 2, 3], n, p=[0.2, 0.5, 0.3])
    loan_sanctioned = np.random.binomial(1, 0.3, n)
    offer_type = np.random.choice([0, 1, 2], n, p=[0.2, 0.4, 0.4])
    cas_issued = np.random.binomial(1, np.where(offer_type == 2, 0.7, 0.05))

    # --- NEW ENHANCED FEATURES ---
    # Study Gap: Higher for older or lower-GPA students
    study_gap = np.random.choice([0, 1, 2, 3, 5], n, p=[0.5, 0.2, 0.1, 0.1, 0.1])
    # Visa Refusal: 10% chance generally
    visa_refusal = np.random.binomial(1, 0.1, n)
    # Days to Intake: Randomly spread across a 6-month window (180 days)
    days_to_intake = np.random.randint(15, 180, n)
    # Session Time: Higher if cas_issued is 1
    session_time_min = (cas_issued * 15) + np.random.poisson(8, n)
    # Proof of Funds Verified: Higher probability if loan is sanctioned
    pof_verified = np.random.binomial(1, np.where(loan_sanctioned == 1, 0.9, 0.4))

    df = pd.DataFrame({
        'gpa': gpa, 'test_score': test_score, 'uni_tier': uni_tier,
        'loan_sanctioned': loan_sanctioned, 'pof_verified': pof_verified,
        'offer_type': offer_type, 'cas_issued': cas_issued, 'study_gap': study_gap,
        'visa_refusal': visa_refusal, 'days_to_intake': days_to_intake,
        'session_time_min': session_time_min
    })
    
    # Enhanced Intent Score Calculation
    # Balances predictive factors (academic, financial, risk) with engagement metrics
    df['intent_score'] = (
        # Academic factors (30% weight)
        (df['gpa'] / 10.0) * 15 +              # GPA contribution (0-15 points)
        (df['test_score'] / 100) * 15 +        # Test score contribution (0-15 points)
        
        # Financial readiness (15% weight)
        df['loan_sanctioned'] * 7.5 +          # Loan sanctioned (0 or 7.5 points)
        df['pof_verified'] * 7.5 +             # POF verified (0 or 7.5 points)
        
        # Risk factors (20% weight)
        (1 - df['visa_refusal']) * 10 +        # No visa refusal (0 or 10 points)
        (1 - df['study_gap'] / 5.0) * 10 +     # Study continuity (0-10 points)
        
        # Engagement metrics (25% weight) - REDUCED from previous 80%
        df['session_time_min'] * 0.5 +         # Session time (0-15.5 points for 31 min)
        df['cas_issued'] * 10 +                # CAS issued (0 or 10 points)
        
        # Urgency factor (10% weight)
        (1 - df['days_to_intake'] / 180) * 10  # Timeline urgency (0-10 points)
    )
    
    # Clip to 0-100 range
    df['intent_score'] = df['intent_score'].clip(0, 100)
    
    return df

if __name__ == "__main__":
    import os
    df_enhanced = generate_enhanced_data(1000)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(data_dir, 'student_data.csv')
    df_enhanced.to_csv(output_path, index=False)
    print("Enhanced Dataset created successfully with", len(df_enhanced), "records")
    print("Columns:", list(df_enhanced.columns))
    print(f"Saved to: {output_path}")