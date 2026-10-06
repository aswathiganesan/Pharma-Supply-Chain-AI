import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 70)
print("TRAINING KNN MODEL")
print("=" * 70)

# Load training and testing data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create KNN pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    ))
])

# Train
print("\nTraining KNN...")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nKNN Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "models/knn_risk_model.pkl")

print("\nModel saved to:")
print("models/knn_risk_model.pkl")

print("\n" + "=" * 70)
print("KNN TRAINING COMPLETED")
print("=" * 70)