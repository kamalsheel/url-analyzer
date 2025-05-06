import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_model(model_path, scaler_path, selector_path):
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    selector = joblib.load(selector_path)
    return clf, scaler, selector

def preprocess_url(url, scaler, selector):
    data = {
        "url_length": [len(url)],
        "url_dot_count": [url.count(".")],
        "url_slash_count": [url.count("/")]
    }
    df = pd.DataFrame(data)
    X_scaled = scaler.transform(df)
    X_selected = selector.transform(X_scaled)
    return X_selected

def analyze_risk(prediction, y_proba):
    confidence_level = np.max(y_proba) * 100
    risk_factor = 100 - confidence_level

    if prediction == "Good":
        if confidence_level > 75:
            interpretation = "✅ Benign URL"
            risk_label = "Low Risk"
        elif confidence_level > 50:
            interpretation = "⚠️ Possibly Benign — Verify"
            risk_label = "Medium Risk"
        else:
            interpretation = "❌ Suspicious — Investigate"
            risk_label = "High Risk"
    else: 
        if confidence_level > 75:
            interpretation = "❌ Malicious (Confirmed)"
            risk_label = "High Risk"
        elif confidence_level > 50:
            interpretation = "❌ Possibly Malicious"
            risk_label = "Medium Risk"
        else:
            interpretation = "❌ Suspicious — Investigate"
            risk_label = "Low Risk"

    print(f"Confidence Level: {confidence_level:.2f}%, Risk Factor: {risk_factor:.2f}%")
    print(f"Risk Category: {risk_label}")
    print(f"Interpretation: {interpretation}")

def main():
    url = input("Enter the URL to analyze: ")
    model_dir = r"C:\Users\KIIT\OneDrive\Pictures\ml-ids\model"

    model_path = os.path.join(model_dir, 'ids_model_final.joblib')
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    selector_path = os.path.join(model_dir, 'selector.joblib')

    clf, scaler, selector = load_model(model_path, scaler_path, selector_path)
    X_test = preprocess_url(url, scaler, selector)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    prediction_label = "Bad" if y_pred[0] == 1 else "Good"
    print(f"Prediction: {prediction_label}")
    analyze_risk(prediction_label, y_proba)

if __name__ == "__main__":
    main()
