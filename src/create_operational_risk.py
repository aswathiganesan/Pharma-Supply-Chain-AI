import pandas as pd

input_file = "data/real/Parametros de stock.xlsx"
output_file = "data/processed/operational_risk_dataset.csv"

sheet = "ABC-XYZ FC 2023-2024"

print("=" * 80)
print("CREATING OPERATIONAL RISK DATASET")
print("=" * 80)

# ---------------------------------------------------------
# 1. READ THE REAL DATA
# ---------------------------------------------------------

df = pd.read_excel(
    input_file,
    sheet_name=sheet,
    header=2
)

print("\nOriginal shape:", df.shape)

# ---------------------------------------------------------
# 2. REMOVE COMPLETELY EMPTY COLUMNS
# ---------------------------------------------------------

df = df.dropna(axis=1, how="all")

# ---------------------------------------------------------
# 3. RENAME IMPORTANT COLUMNS
# ---------------------------------------------------------

rename_columns = {
    "Producto": "Product",
    "Clasif. ABC": "ABC_Class",
    "Clasif. XYZ": "XYZ_Class",
    "Clasif. ABC/XYZ": "ABC_XYZ_Class",

    "Stock a Junio 2025": "Current_Stock",
    "Meses de inventario": "Inventory_Months",
    "Stock Máximo": "Max_Stock",
    "Punto de repo": "Reorder_Point",

    # Numeric critical stock value
    "Stock Crítico": "Critical_Stock",

    # Actual critical condition
    "(Condición crítica)": "Critical_Condition",

    "Demanada diaria": "Daily_Demand",
    "Tiempo prom. entrega (días)": "Avg_Delivery_Time",
    "Stock de seguridad recalculado": "Safety_Stock",
    "Punto de reorden en base a datos": "Calculated_Reorder_Point",
    "Diferencia relativa (%)": "Reorder_Difference"
}

df = df.rename(columns=rename_columns)

# ---------------------------------------------------------
# 4. SHOW IMPORTANT COLUMNS
# ---------------------------------------------------------

print("\nAvailable important columns:")

important_columns = [
    "Product",
    "ABC_Class",
    "XYZ_Class",
    "ABC_XYZ_Class",
    "Current_Stock",
    "Inventory_Months",
    "Max_Stock",
    "Reorder_Point",
    "Critical_Stock",
    "Critical_Condition",
    "Daily_Demand",
    "Avg_Delivery_Time",
    "Safety_Stock",
    "Calculated_Reorder_Point",
    "Reorder_Difference"
]

for column in important_columns:

    if column in df.columns:
        print("OK   -", column)
    else:
        print("MISS -", column)

# ---------------------------------------------------------
# 5. HANDLE CALCULATED REORDER POINT
# ---------------------------------------------------------

if "Calculated_Reorder_Point" not in df.columns:

    print("\nCalculated_Reorder_Point not found.")
    print("Using Reorder_Point as fallback.")

    if "Reorder_Point" in df.columns:

        df["Calculated_Reorder_Point"] = df[
            "Reorder_Point"
        ]

# ---------------------------------------------------------
# 6. SELECT FEATURES
# ---------------------------------------------------------

features = [
    "Product",
    "ABC_Class",
    "XYZ_Class",
    "ABC_XYZ_Class",
    "Current_Stock",
    "Inventory_Months",
    "Max_Stock",
    "Reorder_Point",
    "Critical_Stock",
    "Critical_Condition",
    "Daily_Demand",
    "Avg_Delivery_Time",
    "Safety_Stock",
    "Calculated_Reorder_Point",
    "Reorder_Difference"
]

available_features = [
    column
    for column in features
    if column in df.columns
]

print("\nAvailable features:")
print(available_features)

df = df[available_features].copy()

# ---------------------------------------------------------
# 7. REMOVE EMPTY PRODUCT ROWS
# ---------------------------------------------------------

if "Product" in df.columns:

    df = df.dropna(
        subset=["Product"]
    )

# ---------------------------------------------------------
# 8. CONVERT NUMERIC COLUMNS
# ---------------------------------------------------------

numeric_columns = [
    "Current_Stock",
    "Inventory_Months",
    "Max_Stock",
    "Reorder_Point",
    "Critical_Stock",
    "Daily_Demand",
    "Avg_Delivery_Time",
    "Safety_Stock",
    "Calculated_Reorder_Point",
    "Reorder_Difference"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

# ---------------------------------------------------------
# 9. CREATE OPERATIONAL RISK LEVEL
# ---------------------------------------------------------

def assign_operational_risk(row):

    risk_score = 0

    # ---------------------------------------------
    # ACTUAL CRITICAL CONDITION
    # ---------------------------------------------

    if "Critical_Condition" in row.index:

        if pd.notna(row["Critical_Condition"]):

            status = str(
                row["Critical_Condition"]
            ).strip().lower()

            if status == "crítico":

                risk_score += 3

    # ---------------------------------------------
    # LOW INVENTORY MONTHS
    # ---------------------------------------------

    if "Inventory_Months" in row.index:

        if pd.notna(row["Inventory_Months"]):

            if row["Inventory_Months"] < 1:

                risk_score += 2

            elif row["Inventory_Months"] < 2:

                risk_score += 1

    # ---------------------------------------------
    # CURRENT STOCK BELOW REORDER POINT
    # ---------------------------------------------

    if (
        "Current_Stock" in row.index
        and "Reorder_Point" in row.index
    ):

        current_stock = row["Current_Stock"]
        reorder_point = row["Reorder_Point"]

        if (
            pd.notna(current_stock)
            and pd.notna(reorder_point)
        ):

            if (
                reorder_point > 0
                and current_stock < reorder_point
            ):

                risk_score += 2

    # ---------------------------------------------
    # LOW STOCK RELATIVE TO CRITICAL STOCK
    # ---------------------------------------------

    if (
        "Current_Stock" in row.index
        and "Critical_Stock" in row.index
    ):

        current_stock = row["Current_Stock"]
        critical_stock = row["Critical_Stock"]

        if (
            pd.notna(current_stock)
            and pd.notna(critical_stock)
            and critical_stock > 0
        ):

            if current_stock <= critical_stock:

                risk_score += 2

    # ---------------------------------------------
    # FINAL RISK LEVEL
    # ---------------------------------------------

    if risk_score >= 4:

        return "High"

    elif risk_score >= 2:

        return "Medium"

    else:

        return "Low"


df["Risk_Level"] = df.apply(
    assign_operational_risk,
    axis=1
)

# ---------------------------------------------------------
# 10. SAVE DATASET
# ---------------------------------------------------------

df.to_csv(
    output_file,
    index=False
)

# ---------------------------------------------------------
# 11. DISPLAY RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("OPERATIONAL RISK RESULTS")
print("=" * 80)

print("\nFinal shape:", df.shape)

print("\nRisk Level Distribution:")

print(
    df["Risk_Level"].value_counts()
)

print("\nRisk Level Percentage:")

print(
    (
        df["Risk_Level"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)

print("\nCritical Condition Distribution:")

if "Critical_Condition" in df.columns:

    print(
        df["Critical_Condition"]
        .value_counts(dropna=False)
    )

print("\nFirst 10 records:")

print(
    df.head(10).to_string(
        index=False
    )
)

print("\nSaved to:")
print(output_file)

print("\n" + "=" * 80)
print("OPERATIONAL RISK DATASET CREATED SUCCESSFULLY")
print("=" * 80)