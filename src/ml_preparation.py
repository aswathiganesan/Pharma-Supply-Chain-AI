import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load cleaned dataset
input_file = "data/processed/pharma_supply_chain_cleaned.csv"

df = pd.read_csv(input_file)

print("=" * 60)
print("MACHINE LEARNING DATA PREPARATION")
print("=" * 60)

# --------------------------------------------------
# 1. Separate features and target
# --------------------------------------------------

X = df.drop(columns=["Shipment_ID", "Risk_Level"])
y = df["Risk_Level"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)

# --------------------------------------------------
# 2. Convert categorical columns into numbers
# --------------------------------------------------

X = pd.get_dummies(
    X,
    columns=["Product_Category", "Transportation_Risk"],
    drop_first=True
)

print("\nEncoded Features:")
print(X.columns.tolist())

# --------------------------------------------------
# 3. Convert target labels into numbers
# --------------------------------------------------

target_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

y = y.map(target_mapping)

print("\nEncoded Target:")
print(y.value_counts())

# --------------------------------------------------
# 4. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# 5. Feature scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")

# --------------------------------------------------
# 6. Save prepared data
# --------------------------------------------------

X_train_scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=X.columns
)

X_test_scaled_df = pd.DataFrame(
    X_test_scaled,
    columns=X.columns
)

X_train_scaled_df.to_csv(
    "data/processed/X_train.csv",
    index=False
)

X_test_scaled_df.to_csv(
    "data/processed/X_test.csv",
    index=False
)

y_train.to_csv(
    "data/processed/y_train.csv",
    index=False
)

y_test.to_csv(
    "data/processed/y_test.csv",
    index=False
)

print("\nPrepared datasets saved successfully.")

print("\n" + "=" * 60)
print("MACHINE LEARNING DATA PREPARATION COMPLETED")
print("=" * 60)