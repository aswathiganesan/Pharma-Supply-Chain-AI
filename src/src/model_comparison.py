import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("=" * 70)
print("SVM VS KNN MODEL COMPARISON")
print("=" * 70)

# Load test data
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# Load trained models
svm_model = joblib.load("models/svm_risk_model.pkl")
knn_model = joblib.load("models/knn_risk_model.pkl")

# Predictions
svm_pred = svm_model.predict(X_test)
knn_pred = knn_model.predict(X_test)

models = {
    "SVM": svm_pred,
    "KNN": knn_pred
}

results = []

for name, predictions in models.items():

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, average="macro"
        ),
        "Recall": recall_score(
            y_test, predictions, average="macro"
        ),
        "F1_Score": f1_score(
            y_test, predictions, average="macro"
        )
    })

comparison = pd.DataFrame(results)

print("\nMODEL COMPARISON:")
print(comparison.to_string(index=False))

# Save comparison
comparison.to_csv(
    "data/processed/model_comparison.csv",
    index=False
)

print("\nSaved comparison to:")
print("data/processed/model_comparison.csv")

# Create graph
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score"
]

x = range(len(metrics))

plt.figure(figsize=(10, 6))

plt.plot(
    x,
    comparison.loc[0, metrics],
    marker="o",
    label="SVM"
)

plt.plot(
    x,
    comparison.loc[1, metrics],
    marker="o",
    label="KNN"
)

plt.xticks(x, metrics)
plt.ylim(0, 1.05)

plt.xlabel("Evaluation Metrics")
plt.ylabel("Score")
plt.title("SVM vs KNN Performance Comparison")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "visualizations/model_comparison.png",
    dpi=300
)

plt.close()

print("\nGraph saved to:")
print("visualizations/model_comparison.png")

print("\n" + "=" * 70)
print("MODEL COMPARISON COMPLETED")
print("=" * 70)