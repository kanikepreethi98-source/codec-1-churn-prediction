import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

DATA_PATH = "dataset/customer_churn.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    return df

def preprocess_data(df):
    df = df.copy()
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])
    multi_cols = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
                  'Contract','PaymentMethod']
    df = pd.get_dummies(df, columns=[c for c in multi_cols if c in df.columns], drop_first=True)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    print(f"Features: {X.shape[1]} | Class dist: {dict(y.value_counts())}")
    return X, y, list(X.columns)

def main():
    df = load_data(DATA_PATH)
    X, y, feature_columns = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=RANDOM_STATE, class_weight='balanced'),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE, class_weight='balanced'),
    }
    best_model, best_f1, best_name = None, 0, ""
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name}: Accuracy={acc:.4f} F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_model, best_name = f1, clf, name
    print(f"\nBest model: {best_name}")
    print(classification_report(y_test, best_model.predict(X_test)))
    joblib.dump(best_model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(feature_columns, "feature_columns.pkl")
    print("Artifacts saved. Run: streamlit run app.py")

if __name__ == "__main__":
    main()
