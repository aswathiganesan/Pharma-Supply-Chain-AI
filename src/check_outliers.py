import pandas as pd

file = "data/processed/pharma_delivery_cleaned.csv"

df = pd.read_csv(file)

print("=" * 70)
print("CHECKING EXTREME WAITING-TIME VALUES")
print("=" * 70)

outliers = df[df["Tiempo de espera"] > 100].copy()

print("\nNumber of suspicious records:", len(outliers))

print("\nSuspicious records:")

print(
    outliers[
        [
            "producto",
            "tipo_compra",
            "Fecha de solicitud",
            "fecha_ingreso",
            "fecha_objetivo",
            "Desfase respecto al mes de uso programado",
            "Tiempo de espera"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("OUTLIER CHECK COMPLETED")
print("=" * 70)