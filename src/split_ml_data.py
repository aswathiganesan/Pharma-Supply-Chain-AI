import pandas as pd
from sklearn.model_selection import train_test_split

input_file = "data/processed/ml_features.csv"

df = pd.read_csv(input_file)

print("=" * 70)
print("SPLITTING ML DATA")
print("=" * 70)

X = df.drop("Risk_Level", axis=1)
y = df["Risk_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTotal samples:", len(df))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# Save files
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("\nSaved:")
print("X_train.csv")
print("X_test.csv")
print("y_train.csv")
print("y_test.csv")

print("\n" + "=" * 70)
print("DATA SPLIT COMPLETED")
print("=" * 70)