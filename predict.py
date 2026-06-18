import joblib
import pandas as pd
import os

MODEL_PATH    = "model.pkl"
SCALER_PATH   = "scaler.pkl"
FEATURES_PATH = "feature_columns.pkl"

def load_model_artifacts():
    missing = [p for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH] if not os.path.exists(p)]
    if missing:
        print(f"Missing: {missing}. Run train_model.py first.")
        return None
    return {
        'model':    joblib.load(MODEL_PATH),
        'scaler':   joblib.load(SCALER_PATH),
        'features': joblib.load(FEATURES_PATH),
    }

def preprocess_single(customer, feature_columns):
    df = pd.DataFrame([customer])
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    for col in ['gender','Partner','Dependents','PhoneService','PaperlessBilling']:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(0)
    multi_cols = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
                  'Contract','PaymentMethod']
    df = pd.get_dummies(df, columns=[c for c in multi_cols if c in df.columns], drop_first=True)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]

def get_risk_level(prob):
    if prob >= 0.70: return "High"
    if prob >= 0.45: return "Medium"
    return "Low"

def predict_churn(customer, artifacts):
    X = preprocess_single(customer, artifacts['features'])
    num_cols = [c for c in ['tenure','MonthlyCharges','TotalCharges'] if c in X.columns]
    X[num_cols] = artifacts['scaler'].transform(X[num_cols])
    proba = artifacts['model'].predict_proba(X)[0][1]
    return {
        'churn':       bool(artifacts['model'].predict(X)[0]),
        'probability': float(proba),
        'risk_level':  get_risk_level(proba),
        'confidence':  float(max(proba, 1 - proba)),
    }
