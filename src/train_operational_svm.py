import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("=" * 80)
print("TRAINING OPERATIONAL RISK - SVM")
print("=" * 80)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

X_train = pd.read_csv(
    "data/processed/operational_X_train.csv"
)

X_test = pd.read_csv(
    "data/processed/operational_X_test.csv"
)

y_train = pd.read_csv(
    "data/processed/operational_y_train.csv"
).squeeze()

y_test = pd.read_csv(
    "data/processed/operational_y_test.csv"
).squeeze()

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# ---------------------------------------------------------
# 2. CREATE SVM PIPELINE
# ---------------------------------------------------------

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "svm",
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced"
        )
    )
])

# ---------------------------------------------------------
# 3. TRAIN
# ---------------------------------------------------------

print("\nTraining SVM...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")

# ---------------------------------------------------------
# 4. PREDICT
# ---------------------------------------------------------

y_pred = model.predict(X_test)

# ---------------------------------------------------------
# 5. ACCURACY
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(f"{accuracy:.4f}")

print(
    f"Accuracy Percentage: {accuracy * 100:.2f}%"
)

# ---------------------------------------------------------
# 6. CLASSIFICATION REPORT
# ---------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=["High", "Medium", "Low"],
        zero_division=0
    )
)

# ---------------------------------------------------------
# 7. CONFUSION MATRIX
# ---------------------------------------------------------

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=["High", "Medium", "Low"]
    )
)

# ---------------------------------------------------------
# 8. SAVE MODEL
# ---------------------------------------------------------

output_file = "models/operational_svm.pkl"

joblib.dump(
    model,
    output_file
)

print("\nModel saved to:")
print(output_file)

print("\n" + "=" * 80)
print("SVM TRAINING COMPLETED")
print("=" * 80)