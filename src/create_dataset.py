import pandas as pd
import numpy as np

# Set random seed so the results are reproducible
np.random.seed(42)

# Number of records
n = 500

# Create dataset
data = {
    "Shipment_ID": [f"S{i:04d}" for i in range(1, n + 1)],

    "Product_Category": np.random.choice(
        ["Tablet", "Capsule", "Injection", "Syrup", "Vaccine"],
        n
    ),

    "Supplier_Reliability": np.random.randint(50, 101, n),

    "Inventory_Level": np.random.randint(50, 1001, n),

    "Demand": np.random.randint(100, 1001, n),

    "Lead_Time": np.random.randint(2, 21, n),

    "Order_Quantity": np.random.randint(50, 1001, n),

    "Delivery_Delay": np.random.randint(0, 16, n),

    "Storage_Temperature": np.random.uniform(2, 30, n).round(2),

    "Transportation_Risk": np.random.choice(
        ["Low", "Medium", "High"],
        n
    ),

    "Historical_Delays": np.random.randint(0, 16, n)
}

df = pd.DataFrame(data)


# --------------------------------------------------
# Create Risk Level
# --------------------------------------------------

risk_score = (
    (100 - df["Supplier_Reliability"]) * 0.25
    + df["Lead_Time"] * 1.5
    + df["Delivery_Delay"] * 2
    + df["Historical_Delays"] * 1.5
    + df["Transportation_Risk"].map({
        "Low": 0,
        "Medium": 10,
        "High": 20
    })
    + np.maximum(
        df["Demand"] - df["Inventory_Level"], 0
    ) * 0.02
)

# Convert score into risk categories
df["Risk_Level"] = pd.cut(
    risk_score,
    bins=[-np.inf, 25, 50, np.inf],
    labels=["Low", "Medium", "High"]
)

# Save dataset
output_path = "data/raw/pharma_supply_chain.csv"

df.to_csv(output_path, index=False)

print("==========================================")
print("DATASET CREATED SUCCESSFULLY")
print("==========================================")
print(f"Number of records: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print(f"Saved to: {output_path}")

print("\nFirst 5 records:")
print(df.head())

print("\nRisk distribution:")
print(df["Risk_Level"].value_counts())