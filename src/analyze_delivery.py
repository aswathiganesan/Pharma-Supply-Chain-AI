import pandas as pd

print("=" * 70)
print("ANALYZE DELIVERY SCRIPT STARTED")
print("=" * 70)

# Load cleaned data
file = "data/processed/pharma_delivery_cleaned.csv"

df = pd.read_csv(file)

print("\nDataset shape:")
print(df.shape)

# Waiting time analysis
print("\n" + "-" * 70)
print("WAITING TIME ANALYSIS")
print("-" * 70)

print("Minimum waiting time:", df["Tiempo de espera"].min())
print("Maximum waiting time:", df["Tiempo de espera"].max())
print("Mean waiting time:", round(df["Tiempo de espera"].mean(), 2))
print("Median waiting time:", df["Tiempo de espera"].median())

print("\nWaiting-time percentiles:")
print(
    df["Tiempo de espera"].quantile(
        [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

# Waiting time ranges
print("\n" + "-" * 70)
print("WAITING TIME RANGES")
print("-" * 70)

ranges = pd.cut(
    df["Tiempo de espera"],
    bins=[0, 15, 30, 45, 60, 90, 180, float("inf")],
    labels=[
        "0-15",
        "16-30",
        "31-45",
        "46-60",
        "61-90",
        "91-180",
        "180+"
    ]
)

print(ranges.value_counts().sort_index())

# Schedule deviation
print("\n" + "-" * 70)
print("SCHEDULE DEVIATION ANALYSIS")
print("-" * 70)

print(
    df["Desfase respecto al mes de uso programado"].describe()
)

# Purchase types
print("\n" + "-" * 70)
print("PURCHASE TYPES")
print("-" * 70)

print(df["tipo_compra"].value_counts())

# Longest waiting times
print("\n" + "-" * 70)
print("TOP 20 LONGEST WAITING TIMES")
print("-" * 70)

top_records = df.sort_values(
    "Tiempo de espera",
    ascending=False
).head(20)

print(
    top_records[
        [
            "producto",
            "tipo_compra",
            "Tiempo de espera",
            "Desfase respecto al mes de uso programado"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("DELIVERY ANALYSIS COMPLETED")
print("=" * 70)