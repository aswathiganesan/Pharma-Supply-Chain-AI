import pandas as pd
import joblib

print("=" * 70)
print("PHARMA SUPPLY CHAIN RISK PREDICTION")
print("=" * 70)

# Load trained KNN model
model = joblib.load("models/knn_risk_model.pkl")

# Get user input
schedule_deviation = float(
    input("Enter schedule deviation (days): ")
)

entry_month = int(
    input("Enter entry month (1-12): ")
)

entry_year = int(
    input("Enter entry year: ")
)

target_month = int(
    input("Enter target month (1-12): ")
)

print("\nPurchase Types:")
print("1 - CENABAST")
print("2 - LICITACION")
print("3 - COMPRA AGIL")

purchase_choice = int(
    input("Enter purchase type (1-3): ")
)

# Match the category coding used during training
purchase_codes = {
    1: 0,
    2: 1,
    3: 2
}

purchase_type_code = purchase_codes.get(purchase_choice)

if purchase_type_code is None:
    print("\nInvalid purchase type.")
    exit()

# Create input DataFrame
input_data = pd.DataFrame({
    "Desfase respecto al mes de uso programado": [
        schedule_deviation
    ],
    "entry_month": [
        entry_month
    ],
    "entry_year": [
        entry_year
    ],
    "target_month": [
        target_month
    ],
    "purchase_type_code": [
        purchase_type_code
    ]
})

# Predict
prediction = model.predict(input_data)[0]

print("\n" + "=" * 70)
print("PREDICTED RISK LEVEL:", prediction)
print("=" * 70)