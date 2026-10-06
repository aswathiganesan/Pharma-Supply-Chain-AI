import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("=" * 80)
print("TRAINING OPERATIONAL RISK - RANDOM FOREST")
print("=" * 80)

# ---------------------------------------------------------
# 1. LOAD TRAINING AND TESTING DATA
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
# 2. CREATE RANDOM FOREST MODEL
# ---------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# ---------------------------------------------------------
# 3. TRAIN
# ---------------------------------------------------------

print("\nTraining Random Forest...")

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
# 8. FEATURE IMPORTANCE
# ---------------------------------------------------------

print("\nFeature Importance:")

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(
    importance.to_string(index=False)
)

# ---------------------------------------------------------
# 9. SAVE MODEL
# ---------------------------------------------------------

output_file = "models/operational_random_forest.pkl"

joblib.dump(
    model,
    output_file
)

print("\nModel saved to:")
print(output_file)

print("\n" + "=" * 80)
print("RANDOM FOREST TRAINING COMPLETED")
print("=" * 80)
