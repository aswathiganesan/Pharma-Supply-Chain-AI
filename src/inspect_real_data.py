import pandas as pd
import os

folder = "data/real"

print("=" * 70)
print("PHARMACEUTICAL SUPPLY CHAIN - REAL DATA INSPECTION")
print("=" * 70)

files = [f for f in os.listdir(folder) if f.endswith(".xlsx")]

for file in files:

    path = os.path.join(folder, file)

    print("\n" + "=" * 70)
    print("FILE:", file)
    print("=" * 70)

    excel = pd.ExcelFile(path)

    print("\nSheets:")
    for sheet in excel.sheet_names:
        print(" -", sheet)

        df = pd.read_excel(path, sheet_name=sheet)

        print("   Rows:", df.shape[0])
        print("   Columns:", df.shape[1])

        print("   Column names:")
        for column in df.columns:
            print("    -", column)

        print("\n   First 3 rows:")
        print(df.head(3))

print("\n" + "=" * 70)
print("DATA INSPECTION COMPLETED")
print("=" * 70)