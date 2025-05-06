import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

dataset_paths = [
    r"C:\Users\KIIT\OneDrive\Pictures\ml-ids\dataset\dataset_1.csv",
    r"C:\Users\KIIT\OneDrive\Pictures\ml-ids\dataset\dataset_2.csv",
    r"C:\Users\KIIT\OneDrive\Pictures\ml-ids\dataset\dataset_3.csv",
    r"C:\Users\KIIT\OneDrive\Pictures\ml-ids\dataset\dataset_4.csv",
    r"C:\Users\KIIT\OneDrive\Pictures\ml-ids\dataset\dataset_5.csv",
    r"C:\Users\KIIT\OneDrive\Pictures\ml-ids\dataset\dataset_6.csv",
    r"C:\Users\KIIT\OneDrive\Pictures\ml-ids\dataset\dataset_7.csv"
]

save_directory = r"C:\Users\KIIT\OneDrive\Pictures\ml-ids\model"
os.makedirs(save_directory, exist_ok=True)

dfs = []
for path in dataset_paths:
    try:
        df = pd.read_csv(path)
        dfs.append(df)
    except Exception as e:
        print(f"Error loading {path}: {e}")

df = pd.concat(dfs, ignore_index=True)

df = df[['url', 'type']]
df.fillna("", inplace=True)
df["url"] = df["url"].astype(str)
df = df[df["type"].isin(["good", "bad"])]

label_mapping = {"bad": 0, "good": 1}
df["label"] = df["type"].map(label_mapping)

# Feature engineering
df["url_length"] = df["url"].apply(len)
df["url_dot_count"] = df["url"].apply(lambda x: x.count("."))
df["url_slash_count"] = df["url"].apply(lambda x: x.count("/"))

# Define features and target
X = df.drop(columns=['label', 'type', 'url'])
y = df['label']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=3)
X_selected = selector.fit_transform(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train model
epochs = int(input("Enter the number of epochs: "))
clf = RandomForestClassifier(n_estimators=100, random_state=42)

start_time = time.time()

for epoch in range(epochs):
    epoch_start = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    epoch_end = time.time()

    epoch_time = epoch_end - epoch_start
    total_elapsed = epoch_end - start_time
    estimated_total_time = epoch_time * epochs
    estimated_remaining_time = estimated_total_time - total_elapsed

    print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {accuracy:.2f}")
    print(f"Time elapsed: {total_elapsed:.2f}s, Time per epoch: {epoch_time:.2f}s, Estimated remaining time: {estimated_remaining_time:.2f}s")

    
    if (epoch + 1) % 10 == 0:
        model_path = os.path.join(save_directory, f'ids_model_epoch_{epoch + 1}.joblib')
        joblib.dump(clf, model_path)
        print(f"Model saved at epoch {epoch + 1}: {model_path}")


print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Save final model and preprocessors
joblib.dump(clf, os.path.join(save_directory, 'ids_model_final.joblib'))
joblib.dump(scaler, os.path.join(save_directory, 'scaler.joblib'))
joblib.dump(selector, os.path.join(save_directory, 'selector.joblib'))

print(f"Final model and preprocessors saved in: {save_directory}")