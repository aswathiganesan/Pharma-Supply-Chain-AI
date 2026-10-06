import pandas as pd

input_file = "data/processed/pharma_delivery_cleaned.csv"
output_file = "data/processed/pharma_delivery_cleaned_v2.csv"

df = pd.read_csv(input_file)

print("=" * 70)
print("FIXING DELIVERY OUTLIERS")
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

# Find suspicious waiting-time values
outliers = df["Tiempo de espera"] > 100

print("\nNumber of outliers found:", outliers.sum())

# Correct waiting time using Entry Date - Target Date
df.loc[outliers, "Tiempo de espera"] = (
    df.loc[outliers, "fecha_ingreso"]
    - df.loc[outliers, "fecha_objetivo"]
).dt.days

# Save corrected dataset
df.to_csv(output_file, index=False)

print("\nCorrected outliers:")
print(
    df.loc[
        outliers,
        [
            "producto",
            "tipo_compra",
            "fecha_ingreso",
            "fecha_objetivo",
            "Desfase respecto al mes de uso programado",
            "Tiempo de espera"
        ]
    ].to_string(index=False)
)

print("\nCorrected dataset shape:", df.shape)

print("\nSaved to:")
print(output_file)

print("\n" + "=" * 70)
print("OUTLIER CORRECTION COMPLETED")
print("=" * 70)
