"""
Acolyte Predictive Student Intelligence - Modular Training Script

This script implements a two-step predictive model:
1. Step 1: Visa Approval Score Predictor (RandomForestRegressor)
2. Step 2: Relocation Readiness Classifier (RandomForestClassifier)

The output of Step 1 becomes an input feature for Step 2 (stacked approach).
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath='../data/student_data.csv'):
    """Load the student dataset"""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} records")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"✓ Shape: {df.shape}")
    
    return df


def create_visa_success_score(df):
    """
    Create a theoretical visa success score based on key factors.
    This serves as the target for Step 1 (Visa Scorer).
    
    Score is calculated with strict requirements:
    - Both GPA and test_score must be reasonable (not zero/very low)
    - Previous visa refusal heavily penalizes the score
    - Study gap reduces score proportionally
    """
    print("\n" + "=" * 60)
    print("CREATING VISA SUCCESS SCORE")
    print("=" * 60)
    
    # Base score calculation
    df['visa_success_score'] = (
        (df['gpa'] / 10.0) * 25 +  # Normalize GPA from 10-point scale to 25 points
        (df['test_score'] / 100) * 25 +  # Normalize test score to 25 points
        (1 - df['visa_refusal']) * 25 +  # No refusal = 25 points (increased from 20)
        (1 - (df['study_gap'] / df['study_gap'].max())) * 15 +  # Less gap = more points
        df['loan_sanctioned'] * 10 +  # Loan sanctioned = 10 points (increased back)
        df['pof_verified'] * 5  # POF verified = 5 points (increased back)
    )
    
    # Apply penalties for unrealistic cases
    # CRITICAL: Zero or near-zero test score - extremely severe penalty
    df.loc[df['test_score'] <= 10, 'visa_success_score'] *= 0.15
    
    # Severe penalty if test score is very low (< 50)
    df.loc[(df['test_score'] > 10) & (df['test_score'] < 50), 'visa_success_score'] *= 0.25
    
    # Severe penalty if GPA is very low (< 6.0 on 10-point scale)
    df.loc[df['gpa'] < 6.0, 'visa_success_score'] *= 0.45
    
    # Additional penalty if BOTH are low
    df.loc[(df['test_score'] < 50) & (df['gpa'] < 6.0), 'visa_success_score'] *= 0.15
    
    # Study gap penalty - more balanced progressive penalty
    df.loc[df['study_gap'] >= 3, 'visa_success_score'] *= 0.90  # 10% reduction for 3+ years
    df.loc[df['study_gap'] >= 5, 'visa_success_score'] *= 0.85  # Additional 15% for 5+ years (total ~23.5%)
    
    # HARD THRESHOLDS - Cap maximum score for critical failures
    # These ensure that certain combinations can NEVER result in high visa likelihood
    
    # Zero test score: Maximum 25% visa likelihood regardless of other factors
    df.loc[df['test_score'] == 0, 'visa_success_score'] = np.minimum(
        df.loc[df['test_score'] == 0, 'visa_success_score'], 25
    )
    
    # Test score < 30: Maximum 40% visa likelihood
    df.loc[df['test_score'] < 30, 'visa_success_score'] = np.minimum(
        df.loc[df['test_score'] < 30, 'visa_success_score'], 40
    )
    
    # GPA < 5.0: Maximum 35% visa likelihood
    df.loc[df['gpa'] < 5.0, 'visa_success_score'] = np.minimum(
        df.loc[df['gpa'] < 5.0, 'visa_success_score'], 35
    )
    
    # Both test_score < 50 AND gpa < 6.0: Maximum 45% visa likelihood
    df.loc[(df['test_score'] < 50) & (df['gpa'] < 6.0), 'visa_success_score'] = np.minimum(
        df.loc[(df['test_score'] < 50) & (df['gpa'] < 6.0), 'visa_success_score'], 45
    )
    
    print(f"✓ Visa Success Score created")
    print(f"  Range: {df['visa_success_score'].min():.2f} - {df['visa_success_score'].max():.2f}")
    print(f"  Mean: {df['visa_success_score'].mean():.2f}")
    
    return df


def train_visa_scorer(df):
    """
    Step 1: Train Visa Approval Score Predictor (Regression)
    
    Features: gpa, test_score, study_gap, loan_sanctioned, pof_verified, uni_tier, visa_refusal
    Target: visa_success_score
    """
    print("\n" + "=" * 60)
    print("STEP 1: TRAINING VISA APPROVAL SCORE PREDICTOR")
    print("=" * 60)
    
    # Define features and target
    visa_features = ['gpa', 'test_score', 'study_gap', 'loan_sanctioned', 
                     'pof_verified', 'uni_tier', 'visa_refusal']
    
    X = df[visa_features]
    y = df['visa_success_score']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    
    # Train model
    print("\n⏳ Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\n✓ Model trained successfully!")
    print(f"\n📊 EVALUATION METRICS:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Mean Target Value: {y_test.mean():.4f}")
    print(f"  Relative Error: {(rmse/y_test.mean())*100:.2f}%")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': visa_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📈 FEATURE IMPORTANCE:")
    for idx, row in feature_importance.iterrows():
        print(f"  {row['feature']:20s}: {row['importance']:.4f}")
    
    return model, X, y


def train_relocation_classifier(df, visa_model, X_visa):
    """
    Step 2: Train Relocation Readiness Classifier
    
    Features: visa_approval_score (from Step 1), cas_issued, session_time_min, days_to_intake
    Target: high_relocation_readiness (binary: intent_score > median)
    """
    print("\n" + "=" * 60)
    print("STEP 2: TRAINING RELOCATION READINESS CLASSIFIER")
    print("=" * 60)
    
    # Generate visa approval scores using Step 1 model
    df['visa_approval_score'] = visa_model.predict(X_visa)
    
    # Create binary target: high relocation readiness
    median_intent = df['intent_score'].median()
    df['high_relocation_readiness'] = (df['intent_score'] > median_intent).astype(int)
    
    print(f"✓ Median intent score: {median_intent:.2f}")
    print(f"✓ Binary target created (high_relocation_readiness)")
    
    # ENHANCEMENT: Increase visa score importance by creating weighted feature
    # This makes relocation readiness depend more heavily on visa approval likelihood
    df['weighted_visa_score'] = df['visa_approval_score'] * 3  # Triple the weight
    
    # Define features and target
    # Include both original and weighted visa score for maximum impact
    relocation_features = [
        'visa_approval_score',      # Original visa score
        'weighted_visa_score',      # 3x weighted visa score (increases importance)
        'cas_issued', 
        'session_time_min', 
        'days_to_intake'
    ]
    
    X = df[relocation_features]
    y = df['high_relocation_readiness']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    print(f"\n📊 Class distribution:")
    print(f"  Low Readiness (0): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.1f}%)")
    print(f"  High Readiness (1): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.1f}%)")
    
    # Train model
    print("\n⏳ Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✓ Model trained successfully!")
    print(f"\n📊 EVALUATION METRICS:")
    print(f"  Accuracy: {accuracy:.4f}")
    
    print(f"\n📋 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Low Readiness', 'High Readiness']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"📊 CONFUSION MATRIX:")
    print(f"  True Negatives:  {cm[0][0]:4d}  |  False Positives: {cm[0][1]:4d}")
    print(f"  False Negatives: {cm[1][0]:4d}  |  True Positives:  {cm[1][1]:4d}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': relocation_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📈 FEATURE IMPORTANCE:")
    for idx, row in feature_importance.iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.4f}")
    
    return model


def save_models(visa_model, relocation_model, output_dir='../models'):
    """Save both trained models to disk"""
    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)
    
    # Get absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(os.path.dirname(script_dir), 'models')
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Save Visa Scorer
    visa_path = os.path.join(models_dir, 'visa_scorer.pkl')
    with open(visa_path, 'wb') as f:
        pickle.dump(visa_model, f)
    print(f"✓ Visa Scorer saved to: {visa_path}")
    
    # Save Relocation Classifier
    relocation_path = os.path.join(models_dir, 'relocation_classifier.pkl')
    with open(relocation_path, 'wb') as f:
        pickle.dump(relocation_model, f)
    print(f"✓ Relocation Classifier saved to: {relocation_path}")
    
    print(f"\n✅ All models saved successfully!")


def main():
    """Main training pipeline"""
    print("\n" + "=" * 60)
    print("ACOLYTE PREDICTIVE STUDENT INTELLIGENCE")
    print("Two-Step Model Training Pipeline")
    print("=" * 60)
    
    # Get data path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(script_dir), 'data', 'student_data.csv')
    
    # Load data
    df = load_data(data_path)
    
    # Create visa success score
    df = create_visa_success_score(df)
    
    # Train Step 1: Visa Scorer
    visa_model, X_visa, y_visa = train_visa_scorer(df)
    
    # Train Step 2: Relocation Classifier
    relocation_model = train_relocation_classifier(df, visa_model, X_visa)
    
    # Save models
    save_models(visa_model, relocation_model)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review model performance metrics above")
    print("  2. Create FastAPI backend (app.py)")
    print("  3. Deploy to Render")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
