import pandas as pd

file = "data/processed/pharma_delivery_cleaned.csv"

df = pd.read_csv(file)

outliers = df[df["Tiempo de espera"] > 100].copy()

print("=" * 80)
print("DETAILED OUTLIER DATE CHECK")
print("=" * 80)

for _, row in outliers.iterrows():

    print("\nProduct:", row["producto"])
    print("Purchase Type:", row["tipo_compra"])
    print("Request Date:", row["Fecha de solicitud"])
    print("Entry Date:", row["fecha_ingreso"])
    print("Target Date:", row["fecha_objetivo"])
    print("Schedule Deviation:",
          row["Desfase respecto al mes de uso programado"])
    print("Waiting Time:", row["Tiempo de espera"])

    entry = pd.to_datetime(row["fecha_ingreso"], errors="coerce")
    target = pd.to_datetime(row["fecha_objetivo"], errors="coerce")

    if pd.notna(entry) and pd.notna(target):
        difference = (entry - target).days
        print("Entry - Target Days:", difference)

print("\n" + "=" * 80)
print("CHECK COMPLETED")
print("=" * 80)