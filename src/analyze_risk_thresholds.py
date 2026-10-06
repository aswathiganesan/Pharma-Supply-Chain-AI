import pandas as pd

file = "data/processed/pharma_delivery_cleaned_v2.csv"

df = pd.read_csv(file)

print("=" * 70)
print("ANALYZING RISK THRESHOLDS")
print("=" * 70)

days = df["Tiempo de espera"]

print("\nWaiting Time Statistics:")
print(days.describe())

print("\nPercentiles:")
for p in [10, 25, 50, 60, 70, 75, 80, 85, 90, 95, 99]:
    print(f"{p}th percentile:", days.quantile(p / 100))

print("\nSuggested threshold analysis:")

thresholds = [7, 15, 20, 25, 30, 35, 40, 45, 50]

for threshold in thresholds:
    count = (days > threshold).sum()
    percentage = (count / len(df)) * 100

    print(
        f"More than {threshold:2d} days: "
        f"{count:4d} records ({percentage:.2f}%)"
    )

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS COMPLETED")
print("=" * 70)