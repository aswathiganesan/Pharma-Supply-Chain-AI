import pandas as pd

file = "data/real/Parametros de stock.xlsx"

sheets = [
    "ABC-XYZ FO 2023-2024",
    "ABC-XYZ FC 2023-2024"
]

print("=" * 80)
print("INSPECTING STOCK PARAMETERS")
print("=" * 80)

for sheet in sheets:

    print("\n" + "=" * 80)
    print("SHEET:", sheet)
    print("=" * 80)

    df = pd.read_excel(
        file,
        sheet_name=sheet,
        header=None
    )

    print("\nShape:", df.shape)

    print("\nFirst 10 rows:")
    print(
        df.head(10).to_string(
            index=True,
            header=False
        )
    )

print("\n" + "=" * 80)
print("STOCK PARAMETER INSPECTION COMPLETED")
print("=" * 80)