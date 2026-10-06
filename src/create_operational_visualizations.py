import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)

print("=" * 80)
print("CREATING OPERATIONAL RISK VISUALIZATIONS")
print("=" * 80)

# ---------------------------------------------------------
# CREATE VISUALIZATION FOLDER
# ---------------------------------------------------------

import os

os.makedirs(
    "visualizations",
    exist_ok=True
)

# ---------------------------------------------------------
# LOAD TEST DATA
# ---------------------------------------------------------

X_test = pd.read_csv(
    "data/processed/operational_X_test.csv"
)

y_test = pd.read_csv(
    "data/processed/operational_y_test.csv"
).squeeze()

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------

rf_model = joblib.load(
    "models/operational_random_forest.pkl"
)

knn_model = joblib.load(
    "models/operational_knn.pkl"
)

svm_model = joblib.load(
    "models/operational_svm.pkl"
)

# ---------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------

rf_pred = rf_model.predict(X_test)

knn_pred = knn_model.predict(X_test)

svm_pred = svm_model.predict(X_test)

# ---------------------------------------------------------
# 1. MODEL COMPARISON
# ---------------------------------------------------------

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

knn_accuracy = accuracy_score(
    y_test,
    knn_pred
)

svm_accuracy = accuracy_score(
    y_test,
    svm_pred
)

comparison = pd.DataFrame({
    "Model": [
        "Random Forest",
        "KNN",
        "SVM"
    ],
    "Accuracy": [
        rf_accuracy,
        knn_accuracy,
        svm_accuracy
    ]
})

comparison.to_csv(
    "data/processed/operational_model_comparison.csv",
    index=False
)

plt.figure(figsize=(8, 5))

plt.bar(
    comparison["Model"],
    comparison["Accuracy"] * 100
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.title("Operational Risk Model Comparison")
plt.ylim(0, 100)

for i, value in enumerate(
    comparison["Accuracy"] * 100
):

    plt.text(
        i,
        value + 2,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "visualizations/operational_model_comparison.png",
    dpi=300
)

plt.close()

# ---------------------------------------------------------
# 2. RANDOM FOREST CONFUSION MATRIX
# ---------------------------------------------------------

labels = [
    "High",
    "Medium",
    "Low"
]

cm = confusion_matrix(
    y_test,
    rf_pred,
    labels=labels
)

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title(
    "Random Forest Confusion Matrix"
)

plt.xlabel("Predicted Risk")
plt.ylabel("Actual Risk")

plt.xticks(
    range(len(labels)),
    labels
)

plt.yticks(
    range(len(labels)),
    labels
)

for i in range(len(labels)):

    for j in range(len(labels)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "visualizations/operational_rf_confusion_matrix.png",
    dpi=300
)

plt.close()

# ---------------------------------------------------------
# 3. RANDOM FOREST FEATURE IMPORTANCE
# ---------------------------------------------------------

importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=True
)

importance.to_csv(
    "data/processed/operational_feature_importance.csv",
    index=False
)

plt.figure(figsize=(9, 7))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "visualizations/operational_rf_feature_importance.png",
    dpi=300
)

plt.close()

# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

print("\nModel Accuracy Comparison:")

print(
    comparison.to_string(
        index=False
    )
)

print("\nVisualizations saved:")

print(
    "visualizations/operational_model_comparison.png"
)

print(
    "visualizations/operational_rf_confusion_matrix.png"
)

print(
    "visualizations/operational_rf_feature_importance.png"
)

print("\nData files saved:")

print(
    "data/processed/operational_model_comparison.csv"
)

print(
    "data/processed/operational_feature_importance.csv"
)

print("\n" + "=" * 80)
print("OPERATIONAL VISUALIZATIONS COMPLETED")
print("=" * 80)