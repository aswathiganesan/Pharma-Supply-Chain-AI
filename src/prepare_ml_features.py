import pandas as pd

input_file = "data/processed/pharma_risk_dataset.csv"
output_file = "data/processed/ml_features.csv"

df = pd.read_csv(input_file)

print("=" * 70)
print("PREPARING ML FEATURES")
print("=" * 70)

# Convert dates
df["fecha_ingreso"] = pd.to_datetime(
    df["fecha_ingreso"],
    errors="coerce"
)

df["fecha_objetivo"] = pd.to_datetime(
    df["fecha_objetivo"],
    errors="coerce"
)

# Create useful date-based features
df["entry_month"] = df["fecha_ingreso"].dt.month
df["entry_year"] = df["fecha_ingreso"].dt.year
df["target_month"] = df["fecha_objetivo"].dt.month

# Convert purchase type into numerical categories
df["purchase_type_code"] = (
    df["tipo_compra"].astype("category").cat.codes
)

# Select ML features
# IMPORTANT:
# Tiempo de espera is NOT included because Risk_Level
# was created from it. Including it would cause target leakage.

features = [
    "Desfase respecto al mes de uso programado",
    "entry_month",
    "entry_year",
    "target_month",
    "purchase_type_code"
]

X = df[features].copy()

y = df["Risk_Level"].copy()

# Combine features and target
ml_df = X.copy()
ml_df["Risk_Level"] = y

# Remove missing values
ml_df = ml_df.dropna()

# Save
ml_df.to_csv(output_file, index=False)

print("\nFeatures selected:")
for feature in features:
    print("-", feature)

print("\nFeature dataset shape:", ml_df.shape)

print("\nRisk Level Distribution:")
print(ml_df["Risk_Level"].value_counts())

print("\nFirst 5 rows:")
print(ml_df.head())

print("\nSaved to:")
print(output_file)

print("\n" + "=" * 70)
print("ML FEATURE PREPARATION COMPLETED")
print("=" * 70)