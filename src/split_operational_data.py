import pandas as pd
from sklearn.model_selection import train_test_split

input_file = "data/processed/operational_ml_features.csv"

X_train_file = "data/processed/operational_X_train.csv"
X_test_file = "data/processed/operational_X_test.csv"
y_train_file = "data/processed/operational_y_train.csv"
y_test_file = "data/processed/operational_y_test.csv"

print("=" * 80)
print("SPLITTING OPERATIONAL ML DATA")
print("=" * 80)

# Load data
df = pd.read_csv(input_file)

print("\nDataset shape:", df.shape)

# Separate features and target
X = df.drop(columns=["Risk_Level"])
y = df["Risk_Level"]

# Remove product ID/name
if "Product" in X.columns:
    X = X.drop(columns=["Product"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save files
X_train.to_csv(X_train_file, index=False)
X_test.to_csv(X_test_file, index=False)

y_train.to_csv(y_train_file, index=False)
y_test.to_csv(y_test_file, index=False)

# Display results
print("\nTotal records:", len(df))

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

print("\nTraining percentage:")
print(
    (y_train.value_counts(normalize=True) * 100).round(2)
)

print("\nTesting percentage:")
print(
    (y_test.value_counts(normalize=True) * 100).round(2)
)

print("\nSaved files:")
print(X_train_file)
print(X_test_file)
print(y_train_file)
print(y_test_file)

print("\n" + "=" * 80)
print("OPERATIONAL DATA SPLIT COMPLETED")
print("=" * 80)
