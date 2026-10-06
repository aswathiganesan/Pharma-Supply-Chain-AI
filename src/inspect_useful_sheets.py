import pandas as pd

files = {
    "consumption": (
        "data/real/Consumos 2023-2024.xlsx",
        "Tabla 2. Consumos"
    ),
    "inventory": (
        "data/real/Inventario.xlsx",
        "FO"
    ),
    "inventory_fc": (
        "data/real/Inventario.xlsx",
        "FC"
    ),
    "stock_parameters": (
        "data/real/Parametros de stock.xlsx",
        "FO"
    ),
    "stock_parameters_fc": (
        "data/real/Parametros de stock.xlsx",
        "FC"
    ),
    "wastage": (
        "data/real/Resumen mermas 2023-2024.xlsx",
        "Resumen de Mermas 2023-2024"
    )
}

print("=" * 80)
print("INSPECTING USEFUL PHARMACEUTICAL SUPPLY CHAIN DATA")
print("=" * 80)

for name, (file, sheet) in files.items():

    print("\n" + "=" * 80)
    print("DATASET:", name)
    print("FILE:", file)
    print("SHEET:", sheet)
    print("=" * 80)

    try:
        df = pd.read_excel(file, sheet_name=sheet, header=None)

        print("\nShape:", df.shape)

        print("\nFirst 10 rows:")
        print(df.head(10).to_string(index=True, header=False))

    except Exception as e:
        print("\nERROR:", e)

print("\n" + "=" * 80)
print("INSPECTION COMPLETED")
print("=" * 80)