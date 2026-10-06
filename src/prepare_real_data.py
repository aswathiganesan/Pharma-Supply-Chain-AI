import pandas as pd
import os

print("=" * 70)
print("PHARMACEUTICAL SUPPLY CHAIN - DATA PREPARATION")
print("=" * 70)

# ---------------------------------------------------------
# FILE LOCATION
# ---------------------------------------------------------

input_file = "data/real/Ingresos 2023 -2024 tipos y tiempos de compra.xlsx"

# Read the actual delivery-time sheet
df = pd.read_excel(
    input_file,
    sheet_name="Calculo tiempo de despacho"
)

print("\nOriginal shape:", df.shape)

# ---------------------------------------------------------
# DISPLAY ORIGINAL COLUMNS
# ---------------------------------------------------------

print("\nOriginal columns:")
for column in df.columns:
    print("-", column)

# ---------------------------------------------------------
# SELECT IMPORTANT COLUMNS
# ---------------------------------------------------------

columns_needed = [
    "producto",
    "tipo_compra",
    "Fecha de solicitud",
    "fecha_ingreso",
    "fecha_objetivo",
    "Desfase respecto al mes de uso programado",
    "Tiempo de espera"
]

df = df[columns_needed].copy()

# ---------------------------------------------------------
# REMOVE EMPTY ROWS
# ---------------------------------------------------------

df = df.dropna(how="all")

# Remove rows without product information
df = df.dropna(subset=["producto"])

# ---------------------------------------------------------
# CONVERT DATE COLUMNS
# ---------------------------------------------------------

date_columns = [
    "Fecha de solicitud",
    "fecha_ingreso",
    "fecha_objetivo"
]

for column in date_columns:
    df[column] = pd.to_datetime(
        df[column],
        errors="coerce"
    )

# ---------------------------------------------------------
# CONVERT NUMERIC COLUMNS
# ---------------------------------------------------------

numeric_columns = [
    "Desfase respecto al mes de uso programado",
    "Tiempo de espera"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# ---------------------------------------------------------
# REMOVE INVALID ROWS
# ---------------------------------------------------------

df = df.dropna(
    subset=[
        "Tiempo de espera"
    ]
)

# ---------------------------------------------------------
# BASIC INFORMATION
# ---------------------------------------------------------

print("\nCleaned shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst 10 cleaned rows:")
print(df.head(10))

print("\nStatistics:")
print(
    df[
        [
            "Desfase respecto al mes de uso programado",
            "Tiempo de espera"
        ]
    ].describe()
)

# ---------------------------------------------------------
# SAVE CLEANED DATA
# ---------------------------------------------------------

output_folder = "data/processed"

os.makedirs(
    output_folder,
    exist_ok=True
)

output_file = (
    "data/processed/"
    "pharma_delivery_cleaned.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\nSaved file:")
print(output_file)

print("\n" + "=" * 70)
print("DATA PREPARATION COMPLETED")
print("=" * 70)