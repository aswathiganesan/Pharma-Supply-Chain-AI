import pandas as pd

file = "data/processed/operational_risk_dataset.csv"

df = pd.read_csv(file)

print("=" * 80)
print("OPERATIONAL RISK DATASET ANALYSIS")
print("=" * 80)

print("\nDataset shape:")
print(df.shape)

print("\nRisk Level Count:")
print(df["Risk_Level"].value_counts())

print("\nRisk Level Percentage:")
print(
    (df["Risk_Level"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\n" + "=" * 80)
print("RISK LEVEL BY CRITICAL STOCK CONDITION")
print("=" * 80)

print(
    pd.crosstab(
        df["Critical_Stock"],
        df["Risk_Level"]
    )
)

print("\n" + "=" * 80)
print("RISK LEVEL BY ABC CLASS")
print("=" * 80)

print(
    pd.crosstab(
        df["ABC_Class"],
        df["Risk_Level"]
    )
)

print("\n" + "=" * 80)
print("RISK LEVEL BY XYZ CLASS")
print("=" * 80)

print(
    pd.crosstab(
        df["XYZ_Class"],
        df["Risk_Level"]
    )
)

print("\n" + "=" * 80)
print("AVERAGE FEATURES BY RISK LEVEL")
print("=" * 80)

numeric_features = [
    "Current_Stock",
    "Inventory_Months",
    "Max_Stock",
    "Reorder_Point",
    "Daily_Demand",
    "Avg_Delivery_Time",
    "Safety_Stock",
    "Calculated_Reorder_Point",
    "Reorder_Difference"
]

available = [
    column
    for column in numeric_features
    if column in df.columns
]

print(
    df.groupby("Risk_Level")[available]
    .mean()
    .round(2)
)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETED")
print("=" * 80)