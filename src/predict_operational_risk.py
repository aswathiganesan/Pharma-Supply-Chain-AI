import pandas as pd
import joblib

print("=" * 80)
print("PHARMA SUPPLY CHAIN - OPERATIONAL RISK PREDICTION")
print("=" * 80)

# ---------------------------------------------------------
# 1. LOAD TRAINED RANDOM FOREST MODEL
# ---------------------------------------------------------

model = joblib.load(
    "models/operational_random_forest.pkl"
)

# ---------------------------------------------------------
# 2. GET USER INPUT
# ---------------------------------------------------------

print("\nEnter medicine/supply-chain details:\n")

current_stock = float(
    input("Current Stock: ")
)

inventory_months = float(
    input("Inventory Months: ")
)

max_stock = float(
    input("Maximum Stock: ")
)

reorder_point = float(
    input("Reorder Point: ")
)

critical_stock = float(
    input("Critical Stock: ")
)

daily_demand = float(
    input("Daily Demand: ")
)

avg_delivery_time = float(
    input("Average Delivery Time (days): ")
)

safety_stock = float(
    input("Safety Stock: ")
)

calculated_reorder_point = float(
    input("Calculated Reorder Point: ")
)

reorder_difference = float(
    input("Reorder Difference: ")
)

# ---------------------------------------------------------
# ABC CLASS
# ---------------------------------------------------------

print("\nABC Class:")
print("0 = A")
print("1 = B")
print("2 = C")

abc_code = int(
    input("Enter ABC Code: ")
)

# ---------------------------------------------------------
# XYZ CLASS
# ---------------------------------------------------------

print("\nXYZ Class:")
print("0 = X")
print("1 = Y")
print("2 = Z")

xyz_code = int(
    input("Enter XYZ Code: ")
)

# ---------------------------------------------------------
# 3. CREATE INPUT DATAFRAME
# ---------------------------------------------------------

input_data = pd.DataFrame([{

    "Current_Stock": current_stock,

    "Inventory_Months": inventory_months,

    "Max_Stock": max_stock,

    "Reorder_Point": reorder_point,

    "Critical_Stock": critical_stock,

    "Daily_Demand": daily_demand,

    "Avg_Delivery_Time": avg_delivery_time,

    "Safety_Stock": safety_stock,

    "Calculated_Reorder_Point":
        calculated_reorder_point,

    "Reorder_Difference":
        reorder_difference,

    "ABC_Code": abc_code,

    "XYZ_Code": xyz_code
}])

# ---------------------------------------------------------
# 4. PREDICT RISK
# ---------------------------------------------------------

prediction = model.predict(
    input_data
)[0]

# ---------------------------------------------------------
# 5. PREDICTION PROBABILITY
# ---------------------------------------------------------

probabilities = model.predict_proba(
    input_data
)[0]

classes = model.classes_

probability_table = pd.DataFrame({
    "Risk_Level": classes,
    "Probability": probabilities
})

# ---------------------------------------------------------
# 6. DISPLAY RESULT
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("RISK PREDICTION RESULT")
print("=" * 80)

print(
    f"\nPredicted Operational Risk: {prediction}"
)

print("\nRisk Probabilities:")

for _, row in probability_table.iterrows():

    print(
        f"{row['Risk_Level']}: "
        f"{row['Probability'] * 100:.2f}%"
    )

# ---------------------------------------------------------
# 7. SIMPLE RECOMMENDATION
# ---------------------------------------------------------

print("\nRecommendation:")

if prediction == "High":

    print(
        "URGENT: Review stock levels and "
        "consider immediate replenishment."
    )

elif prediction == "Medium":

    print(
        "WARNING: Monitor inventory closely "
        "and plan replenishment."
    )

else:

    print(
        "LOW RISK: Inventory condition appears "
        "stable. Continue regular monitoring."
    )

print("\n" + "=" * 80)
print("PREDICTION COMPLETED")
print("=" * 80)