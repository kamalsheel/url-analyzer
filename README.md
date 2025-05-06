# url-analyzer
🔍 URL Risk Analyzer – Machine Learning-Based Threat Detection
This project is a lightweight, ML-powered URL risk assessment tool designed to classify URLs as either malicious or benign based on basic structural features.

🚀 Features
Real-time URL classification using a trained ML model (joblib)

Extracts simple yet effective URL features:

Length of URL

Number of dots (.)

Number of slashes (/)

Applies a pre-fitted scaler and feature selector for consistent input processing

Outputs:

Prediction (Good / Bad)

Confidence Level

Risk Factor

Human-readable interpretation (✅ Low Risk, ❌ High Risk, etc.)

🛠️ Tech Stack
scikit-learn

pandas, numpy

joblib (for model serialization)

matplotlib, seaborn (included for possible future visualizations)

📂 Folder Structure
model-

├── ids_model_final.joblib     # Trained classifier

├── scaler.joblib              # Feature scaler

└── selector.joblib            # Feature selector

🔧 How It Works
User inputs a URL.

The program extracts features and preprocesses them using saved scaler and selector.

The model predicts the likelihood of the URL being malicious.

A detailed risk interpretation is printed, including confidence and severity level.

📌 Example Output
yaml
Copy
Edit
Enter the URL to analyze: http://malicious.site/phishing
Prediction: Bad
Confidence Level: 93.76%, Risk Factor: 6.24%
Risk Category: High Risk
Interpretation: ❌ Malicious (Confirmed)
📈 Use Case
This can be integrated into a browser extension, firewall, or endpoint security system to enhance phishing detection, malware prevention, or zero-trust network policies.
