from pathlib import Path

import joblib
import pandas as pd

from django.http import JsonResponse
from django.shortcuts import render


# ============================================================
# BASE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "operational_random_forest.pkl"
)

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "operational_risk_dataset.csv"
)


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

try:

    model = joblib.load(MODEL_PATH)

    print(
        "Random Forest model loaded successfully."
    )

except Exception as e:

    model = None

    print(
        "ERROR: Unable to load Random Forest model:",
        e
    )


# ============================================================
# LOAD OPERATIONAL RISK DATASET
# ============================================================

try:

    product_data = pd.read_csv(DATA_PATH)

    print(
        "Operational risk dataset loaded successfully."
    )

except Exception as e:

    product_data = pd.DataFrame()

    print(
        "ERROR: Unable to load operational risk dataset:",
        e
    )


# ============================================================
# PRODUCT LIST
# ============================================================

if (
    not product_data.empty
    and "Product" in product_data.columns
):

    products = (
        product_data["Product"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

else:

    products = []


# ============================================================
# ABC CLASSIFICATION CONVERSION
# ============================================================

def convert_abc_code(value):

    """
    ABC classification:

    A -> 0
    B -> 1
    C -> 2
    """

    if pd.isna(value):

        return 0

    value = str(value).strip().upper()

    if value == "A":

        return 0

    if value == "B":

        return 1

    if value == "C":

        return 2

    try:

        return int(float(value))

    except Exception:

        return 0


# ============================================================
# XYZ CLASSIFICATION CONVERSION
# ============================================================

def convert_xyz_code(value):

    """
    XYZ classification:

    X -> 0
    Y -> 1
    Z -> 2
    """

    if pd.isna(value):

        return 0

    value = str(value).strip().upper()

    if value == "X":

        return 0

    if value == "Y":

        return 1

    if value == "Z":

        return 2

    try:

        return int(float(value))

    except Exception:

        return 0


# ============================================================
# SAFE NUMERIC VALUE
# ============================================================

def safe_numeric(value, default=0):

    """
    Safely convert a value to float.

    Blank / NaN / invalid values become default.
    """

    if value is None:

        return default

    try:

        if pd.isna(value):

            return default

    except Exception:

        pass

    try:

        return float(value)

    except Exception:

        return default


# ============================================================
# PRODUCT DATA API
# ============================================================

def get_product_data(request):

    # --------------------------------------------------------
    # GET PRODUCT ID
    # --------------------------------------------------------

    product_id = (
        request.GET
        .get("product_id", "")
        .strip()
    )


    # --------------------------------------------------------
    # VALIDATE PRODUCT ID
    # --------------------------------------------------------

    if not product_id:

        return JsonResponse(
            {
                "error": "Product ID is required."
            },
            status=400
        )


    # --------------------------------------------------------
    # CHECK DATASET
    # --------------------------------------------------------

    if product_data.empty:

        return JsonResponse(
            {
                "error": "Product dataset is empty."
            },
            status=500
        )


    # --------------------------------------------------------
    # CHECK PRODUCT COLUMN
    # --------------------------------------------------------

    if "Product" not in product_data.columns:

        return JsonResponse(
            {
                "error":
                    "Product column is missing from dataset."
            },
            status=500
        )


    # --------------------------------------------------------
    # FIND PRODUCT
    # --------------------------------------------------------

    matching_rows = product_data[
        product_data["Product"]
        .astype(str)
        .str.strip()
        .str.lower()
        ==
        product_id.lower()
    ]


    if matching_rows.empty:

        return JsonResponse(
            {
                "error":
                    f"Product {product_id} was not found."
            },
            status=404
        )


    # --------------------------------------------------------
    # FIRST MATCH
    # --------------------------------------------------------

    row = matching_rows.iloc[0]


    # ========================================================
    # GET ABC CLASSIFICATION
    # ========================================================

    if "ABC_Class" in row.index:

        abc_value = row["ABC_Class"]

    elif "ABC_Code" in row.index:

        abc_value = row["ABC_Code"]

    else:

        abc_value = 0


    # ========================================================
    # GET XYZ CLASSIFICATION
    # ========================================================

    if "XYZ_Class" in row.index:

        xyz_value = row["XYZ_Class"]

    elif "XYZ_Code" in row.index:

        xyz_value = row["XYZ_Code"]

    else:

        xyz_value = 0


    # ========================================================
    # GET CALCULATED REORDER POINT
    # ========================================================

    if "Calculated_Reorder_Point" in row.index:

        calculated_reorder_point = safe_numeric(
            row["Calculated_Reorder_Point"]
        )

    elif "Reorder_Point" in row.index:

        calculated_reorder_point = safe_numeric(
            row["Reorder_Point"]
        )

    else:

        calculated_reorder_point = 0


    # ========================================================
    # BUILD RESPONSE
    # ========================================================

    response_data = {

        "Product":
            str(
                row.get(
                    "Product",
                    product_id
                )
            ),


        "Current_Stock":
            safe_numeric(
                row.get(
                    "Current_Stock",
                    0
                )
            ),


        "Inventory_Months":
            safe_numeric(
                row.get(
                    "Inventory_Months",
                    0
                )
            ),


        "Max_Stock":
            safe_numeric(
                row.get(
                    "Max_Stock",
                    0
                )
            ),


        "Reorder_Point":
            safe_numeric(
                row.get(
                    "Reorder_Point",
                    0
                )
            ),


        "Critical_Stock":
            safe_numeric(
                row.get(
                    "Critical_Stock",
                    0
                )
            ),


        "Daily_Demand":
            safe_numeric(
                row.get(
                    "Daily_Demand",
                    0
                )
            ),


        "Avg_Delivery_Time":
            safe_numeric(
                row.get(
                    "Avg_Delivery_Time",
                    0
                )
            ),


        "Safety_Stock":
            safe_numeric(
                row.get(
                    "Safety_Stock",
                    0
                )
            ),


        "Calculated_Reorder_Point":
            calculated_reorder_point,


        "Reorder_Difference":
            safe_numeric(
                row.get(
                    "Reorder_Difference",
                    0
                )
            ),


        "ABC_Code":
            convert_abc_code(
                abc_value
            ),


        "XYZ_Code":
            convert_xyz_code(
                xyz_value
            ),

    }


    # --------------------------------------------------------
    # DEBUG OUTPUT
    # --------------------------------------------------------

    print(
        f"Product API requested: {product_id}"
    )

    print(
        "API response:",
        response_data
    )


    # --------------------------------------------------------
    # RETURN JSON
    # --------------------------------------------------------

    return JsonResponse(
        response_data
    )


# ============================================================
# MAIN PREDICTION VIEW
# ============================================================

def predict_risk(request):

    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    prediction = None

    probabilities = []

    recommendation = ""

    selected_product = ""

    error = None


    # ========================================================
    # FORM VALUES
    # ========================================================

    current_stock = ""

    inventory_months = ""

    max_stock = ""

    reorder_point = ""

    critical_stock = ""

    daily_demand = ""

    avg_delivery_time = ""

    safety_stock = ""

    calculated_reorder_point = ""

    reorder_difference = ""

    abc_code = 0

    xyz_code = 0


    # ========================================================
    # HIGH-RISK PRODUCTS
    # ========================================================

    high_risk_list = []


    if (
        not product_data.empty
        and
        "Risk_Level" in product_data.columns
    ):

        high_risk_products = (

            product_data[

                product_data["Risk_Level"]
                .astype(str)
                .str.strip()
                .str.lower()
                ==
                "high"

            ].copy()

        )


        # ----------------------------------------------------
        # SORT BY CURRENT STOCK
        # ----------------------------------------------------

        if "Current_Stock" in high_risk_products.columns:

            high_risk_products = (
                high_risk_products
                .sort_values(
                    by="Current_Stock",
                    ascending=True
                )
            )


        # ----------------------------------------------------
        # CREATE HIGH-RISK LIST
        # ----------------------------------------------------

        for _, row in high_risk_products.iterrows():

            current_stock_value = safe_numeric(
                row.get(
                    "Current_Stock",
                    0
                )
            )


            reorder_point_value = safe_numeric(
                row.get(
                    "Reorder_Point",
                    0
                )
            )


            critical_stock_value = safe_numeric(
                row.get(
                    "Critical_Stock",
                    0
                )
            )


            # ------------------------------------------------
            # ACTION
            # ------------------------------------------------

            if current_stock_value <= 0:

                action = (
                    "Urgent Stockout"
                )

            elif (
                critical_stock_value > 0
                and
                current_stock_value
                <=
                critical_stock_value
            ):

                action = (
                    "Urgent Replenishment"
                )

            elif (
                reorder_point_value > 0
                and
                current_stock_value
                <
                reorder_point_value
            ):

                action = (
                    "Immediate Replenishment"
                )

            else:

                action = (
                    "Review Stock Level"
                )


            high_risk_list.append(

                {

                    "product":
                        str(
                            row.get(
                                "Product",
                                ""
                            )
                        ),

                    "risk":
                        "HIGH",

                    "action":
                        action,

                }

            )


    # ========================================================
    # POST REQUEST
    # ========================================================

    if request.method == "POST":

        try:

            # =================================================
            # SELECTED PRODUCT
            # =================================================

            selected_product = (
                request.POST
                .get(
                    "product_id",
                    ""
                )
                .strip()
            )


            # =================================================
            # READ FORM VALUES
            # =================================================

            current_stock = safe_numeric(
                request.POST.get(
                    "current_stock",
                    0
                )
            )


            inventory_months = safe_numeric(
                request.POST.get(
                    "inventory_months",
                    0
                )
            )


            max_stock = safe_numeric(
                request.POST.get(
                    "max_stock",
                    0
                )
            )


            reorder_point = safe_numeric(
                request.POST.get(
                    "reorder_point",
                    0
                )
            )


            critical_stock = safe_numeric(
                request.POST.get(
                    "critical_stock",
                    0
                )
            )


            daily_demand = safe_numeric(
                request.POST.get(
                    "daily_demand",
                    0
                )
            )


            avg_delivery_time = safe_numeric(
                request.POST.get(
                    "avg_delivery_time",
                    0
                )
            )


            safety_stock = safe_numeric(
                request.POST.get(
                    "safety_stock",
                    0
                )
            )


            calculated_reorder_point = safe_numeric(
                request.POST.get(
                    "calculated_reorder_point",
                    0
                )
            )


            reorder_difference = safe_numeric(
                request.POST.get(
                    "reorder_difference",
                    0
                )
            )


            try:

                abc_code = int(
                    request.POST.get(
                        "abc_code",
                        0
                    )
                )

            except Exception:

                abc_code = 0


            try:

                xyz_code = int(
                    request.POST.get(
                        "xyz_code",
                        0
                    )
                )

            except Exception:

                xyz_code = 0


            # =================================================
            # MODEL CHECK
            # =================================================

            if model is None:

                raise Exception(
                    "Random Forest model could not be loaded."
                )


            # =================================================
            # MODEL FEATURES
            # =================================================

            feature_names = [

                "Current_Stock",

                "Inventory_Months",

                "Max_Stock",

                "Reorder_Point",

                "Critical_Stock",

                "Daily_Demand",

                "Avg_Delivery_Time",

                "Safety_Stock",

                "Calculated_Reorder_Point",

                "Reorder_Difference",

                "ABC_Code",

                "XYZ_Code",

            ]


            feature_values = [

                current_stock,

                inventory_months,

                max_stock,

                reorder_point,

                critical_stock,

                daily_demand,

                avg_delivery_time,

                safety_stock,

                calculated_reorder_point,

                reorder_difference,

                abc_code,

                xyz_code,

            ]


            model_input = pd.DataFrame(

                [feature_values],

                columns=feature_names

            )


            # =================================================
            # PREDICT
            # =================================================

            prediction_result = model.predict(
                model_input
            )


            prediction = str(
                prediction_result[0]
            )


            # =================================================
            # PREDICT PROBABILITIES
            # =================================================

            if hasattr(
                model,
                "predict_proba"
            ):

                probability_values = (
                    model
                    .predict_proba(
                        model_input
                    )[0]
                )


                if hasattr(
                    model,
                    "classes_"
                ):

                    class_names = (
                        model.classes_
                    )

                else:

                    class_names = [
                        "High",
                        "Medium",
                        "Low"
                    ]


                for (
                    class_name,
                    probability
                ) in zip(
                    class_names,
                    probability_values
                ):

                    probabilities.append(

                        {

                            "risk":
                                str(
                                    class_name
                                ),

                            "probability":
                                round(
                                    float(
                                        probability
                                    )
                                    * 100,
                                    2
                                ),

                        }

                    )


                probabilities = sorted(

                    probabilities,

                    key=lambda item:
                        item["probability"],

                    reverse=True

                )


            # =================================================
            # AI RECOMMENDATION
            # =================================================

            if prediction.lower() == "high":

                # ---------------------------------------------
                # STOCKOUT
                # ---------------------------------------------

                if current_stock <= 0:

                    recommendation = (

                        "URGENT STOCKOUT: Current stock "
                        "is zero. Immediate replenishment "
                        "is required to avoid supply "
                        "disruption."

                    )


                # ---------------------------------------------
                # CRITICAL STOCK
                # ---------------------------------------------

                elif (
                    critical_stock > 0
                    and
                    current_stock
                    <=
                    critical_stock
                ):

                    recommendation = (

                        "URGENT: Current stock is at or "
                        "below the critical stock level. "
                        "Immediate replenishment and "
                        "stock review are recommended."

                    )


                # ---------------------------------------------
                # BELOW REORDER POINT
                # ---------------------------------------------

                elif (
                    reorder_point > 0
                    and
                    current_stock
                    <
                    reorder_point
                ):

                    recommendation = (

                        "URGENT: Current stock is below "
                        "the reorder point. Consider "
                        "immediate replenishment."

                    )


                # ---------------------------------------------
                # LOW INVENTORY COVERAGE
                # ---------------------------------------------

                elif inventory_months < 1:

                    recommendation = (

                        "URGENT: Inventory coverage is "
                        "below one month. Review demand "
                        "and arrange replenishment."

                    )


                # ---------------------------------------------
                # GENERAL HIGH RISK
                # ---------------------------------------------

                else:

                    recommendation = (

                        "URGENT: Review stock levels, "
                        "replenishment requirements, "
                        "and supply-chain conditions."

                    )


            elif prediction.lower() == "medium":

                recommendation = (

                    "MONITOR: Stock conditions require "
                    "regular monitoring. Review demand, "
                    "reorder levels, and replenishment "
                    "planning."

                )


            else:

                recommendation = (

                    "LOW RISK: Current operational "
                    "conditions appear stable. Continue "
                    "regular monitoring."

                )


        except Exception as e:

            error = str(e)

            print(
                "Prediction error:",
                e
            )


    # ========================================================
    # DYNAMIC RISK SUMMARY
    # ========================================================

    low_count = 0

    medium_count = 0

    high_count = 0

    total_count = 0


    if (
        not product_data.empty
        and
        "Risk_Level" in product_data.columns
    ):

        risk_counts = (

            product_data["Risk_Level"]

            .astype(str)

            .str.strip()

            .str.lower()

            .value_counts()

        )


        low_count = int(
            risk_counts.get(
                "low",
                0
            )
        )


        medium_count = int(
            risk_counts.get(
                "medium",
                0
            )
        )


        high_count = int(
            risk_counts.get(
                "high",
                0
            )
        )


        total_count = (

            low_count
            +
            medium_count
            +
            high_count

        )


    # ========================================================
    # RISK PERCENTAGES
    # ========================================================

    low_percentage = 0

    medium_percentage = 0

    high_percentage = 0


    if total_count > 0:

        low_percentage = round(

            (
                low_count
                /
                total_count
            )
            * 100,

            1

        )


        medium_percentage = round(

            (
                medium_count
                /
                total_count
            )
            * 100,

            1

        )


        high_percentage = round(

            (
                high_count
                /
                total_count
            )
            * 100,

            1

        )


    # ========================================================
    # CONTEXT
    # ========================================================

    context = {

        # Product list

        "products":
            products,


        # High-risk products

        "high_risk_products":
            high_risk_list,


        # Risk summary

        "low_count":
            low_count,

        "medium_count":
            medium_count,

        "high_count":
            high_count,

        "total_count":
            total_count,


        # Risk percentages

        "low_percentage":
            low_percentage,

        "medium_percentage":
            medium_percentage,

        "high_percentage":
            high_percentage,


        # Prediction

        "prediction":
            prediction,

        "probabilities":
            probabilities,

        "recommendation":
            recommendation,


        # Selected product

        "selected_product":
            selected_product,


        # Form values

        "current_stock":
            current_stock,

        "inventory_months":
            inventory_months,

        "max_stock":
            max_stock,

        "reorder_point":
            reorder_point,

        "critical_stock":
            critical_stock,

        "daily_demand":
            daily_demand,

        "avg_delivery_time":
            avg_delivery_time,

        "safety_stock":
            safety_stock,

        "calculated_reorder_point":
            calculated_reorder_point,

        "reorder_difference":
            reorder_difference,

        "abc_code":
            abc_code,

        "xyz_code":
            xyz_code,


        # Error

        "error":
            error,

    }


    # ========================================================
    # RENDER PAGE
    # ========================================================

    return render(

        request,

        "risk_prediction/predict.html",

        context

    )