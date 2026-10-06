import pandas as pd

input_file = "data/processed/pharma_delivery_cleaned_v2.csv"
output_file = "data/processed/pharma_risk_dataset.csv"

df = pd.read_csv(input_file)

print("=" * 70)
print("CREATING RISK LEVEL")
print("=" * 70)

# Create Risk_Level based on delivery delay
def assign_risk(days):
    if days <= 25:
        return "Low"
    elif days <= 40:
        return "Medium"
    else:
        return "High"

df["Risk_Level"] = df["Tiempo de espera"].apply(assign_risk)

# Display distribution
print("\nRisk Level Distribution:")
print(df["Risk_Level"].value_counts())

print("\nRisk Level Percentage:")
print(
    (df["Risk_Level"].value_counts(normalize=True) * 100)
    .round(2)
)

# Save final dataset
df.to_csv(output_file, index=False)

print("\nFinal dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nSaved to:")
print(output_file)

print("\n" + "=" * 70)
print("RISK LEVEL CREATION COMPLETED")
print("=" * 70)