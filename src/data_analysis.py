import pandas as pd
import numpy as np

# Load the dataset
file_path = "data/raw/pharma_supply_chain.csv"
df = pd.read_csv(file_path)

print("=" * 60)
print("PHARMA SUPPLY CHAIN DATA ANALYSIS")
print("=" * 60)

# 1. Display first 5 records
print("\n1. FIRST 5 RECORDS")
print(df.head())

# 2. Dataset dimensions
print("\n2. DATASET SHAPE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 3. Column names
print("\n3. COLUMN NAMES")
print(df.columns.tolist())

# 4. Data types
print("\n4. DATA TYPES")
print(df.dtypes)

# 5. Missing values
print("\n5. MISSING VALUES")
print(df.isnull().sum())

# 6. Duplicate records
print("\n6. DUPLICATE RECORDS")
print(df.duplicated().sum())

# 7. Statistical summary
print("\n7. STATISTICAL SUMMARY")
print(df.describe())

# 8. Risk distribution
print("\n8. RISK LEVEL DISTRIBUTION")
print(df["Risk_Level"].value_counts())

# 9. Product category distribution
print("\n9. PRODUCT CATEGORY DISTRIBUTION")
print(df["Product_Category"].value_counts())

# 10. Average values
print("\n10. AVERAGE VALUES")
print("Average Supplier Reliability:",
      np.mean(df["Supplier_Reliability"]))

print("Average Inventory Level:",
      np.mean(df["Inventory_Level"]))

print("Average Demand:",
      np.mean(df["Demand"]))

print("Average Delivery Delay:",
      np.mean(df["Delivery_Delay"]))

print("Average Lead Time:",
      np.mean(df["Lead_Time"]))

print("\n" + "=" * 60)
print("DATA ANALYSIS COMPLETED")
print("=" * 60)