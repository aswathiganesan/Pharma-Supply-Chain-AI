import pandas as pd
import numpy as np

# Load dataset
input_file = "data/raw/pharma_supply_chain.csv"
output_file = "data/processed/pharma_supply_chain_cleaned.csv"

df = pd.read_csv(input_file)

print("=" * 60)
print("PHARMA SUPPLY CHAIN DATA PREPROCESSING")
print("=" * 60)

# --------------------------------------------------
# 1. Display original dataset information
# --------------------------------------------------

print("\nOriginal Dataset Shape:")
print(df.shape)

# --------------------------------------------------
# 2. Check missing values
# --------------------------------------------------

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# --------------------------------------------------
# 3. Remove duplicate records
# --------------------------------------------------

duplicate_count = df.duplicated().sum()

print("\nDuplicate Records:", duplicate_count)

df = df.drop_duplicates()

# --------------------------------------------------
# 4. Handle missing numerical values
# --------------------------------------------------

numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# --------------------------------------------------
# 5. Handle missing categorical values
# --------------------------------------------------

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# --------------------------------------------------
# 6. Check missing values after cleaning
# --------------------------------------------------

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# --------------------------------------------------
# 7. Display cleaned dataset information
# --------------------------------------------------

print("\nCleaned Dataset Shape:")
print(df.shape)

# --------------------------------------------------
# 8. Display categorical columns
# --------------------------------------------------

print("\nCategorical Columns:")
print(categorical_columns.tolist())

# --------------------------------------------------
# 9. Display numerical columns
# --------------------------------------------------

print("\nNumerical Columns:")
print(numeric_columns.tolist())

# --------------------------------------------------
# 10. Save cleaned dataset
# --------------------------------------------------

df.to_csv(output_file, index=False)

print("\nCleaned dataset saved successfully!")
print("File:", output_file)

print("\n" + "=" * 60)
print("DATA PREPROCESSING COMPLETED")
print("=" * 60)