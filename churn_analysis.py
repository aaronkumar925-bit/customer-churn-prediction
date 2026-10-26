#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 23:09:23 2026

@author: aaronkumar.p
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# CUSTOMER CHURN PREDICTION
# ============================================================


# 1. CREATE CUSTOMER DATASET

data = {
    "CustomerID": [
        "C001", "C002", "C003", "C004", "C005",
        "C006", "C007", "C008", "C009", "C010",
        "C011", "C012", "C013", "C014", "C015",
        "C016", "C017", "C018", "C019", "C020",
        "C021", "C022", "C023", "C024", "C025",
        "C026", "C027", "C028", "C029", "C030"
    ],

    "Age": [
        22, 35, 41, 29, 52,
        24, 38, 45, 31, 27,
        50, 33, 40, 26, 48,
        36, 23, 55, 30, 43,
        28, 47, 34, 21, 51,
        39, 25, 44, 32, 46
    ],

    "Tenure_Months": [
        3, 24, 36, 8, 48,
        5, 30, 42, 12, 6,
        60, 20, 34, 4, 55,
        28, 7, 72, 15, 39,
        10, 50, 18, 2, 58,
        32, 9, 45, 14, 52
    ],

    "Monthly_Charges": [
        850, 1200, 1500, 950, 1800,
        700, 1350, 1600, 1100, 900,
        2000, 1250, 1450, 750, 1900,
        1300, 800, 2100, 1050, 1550,
        980, 1750, 1180, 650, 1950,
        1400, 820, 1650, 1080, 1700
    ],

    "Support_Calls": [
        5, 1, 0, 4, 0,
        6, 1, 0, 3, 5,
        0, 2, 1, 6, 0,
        2, 5, 0, 3, 1,
        4, 0, 3, 7, 0,
        1, 4, 0, 3, 1
    ],

    "Contract": [
        "Monthly", "Yearly", "Yearly", "Monthly", "Yearly",
        "Monthly", "Yearly", "Yearly", "Monthly", "Monthly",
        "Yearly", "Yearly", "Yearly", "Monthly", "Yearly",
        "Yearly", "Monthly", "Yearly", "Monthly", "Yearly",
        "Monthly", "Yearly", "Monthly", "Monthly", "Yearly",
        "Yearly", "Monthly", "Yearly", "Monthly", "Yearly"
    ],

    "Churn": [
        1, 0, 0, 1, 0,
        1, 0, 0, 1, 1,
        0, 0, 0, 1, 0,
        0, 1, 0, 1, 0,
        1, 0, 1, 1, 0,
        0, 0, 1, 0, 0
    ]
}

data = pd.DataFrame(data)


# ============================================================
# 2. DISPLAY DATASET
# ============================================================

print("\n========================================")
print("       CUSTOMER CHURN DATASET")
print("========================================")

print(data)


# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

print("\n========================================")
print("       DATASET INFORMATION")
print("========================================")

print("Number of customers:", len(data))
print("Number of columns:", len(data.columns))

print("\nColumn data types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicate records:")
print(data.duplicated().sum())


# ============================================================
# 4. BASIC STATISTICS
# ============================================================

print("\n========================================")
print("       BASIC STATISTICS")
print("========================================")

print(data.describe())


# ============================================================
# 5. CHURN ANALYSIS
# ============================================================

churn_count = data["Churn"].value_counts()

print("\n========================================")
print("       CHURN ANALYSIS")
print("========================================")

print("Customers who stayed:", churn_count.get(0, 0))
print("Customers who churned:", churn_count.get(1, 0))

churn_rate = data["Churn"].mean() * 100

print(f"Churn rate: {churn_rate:.2f}%")


# ============================================================
# 6. CHURN BY CONTRACT TYPE
# ============================================================

contract_churn = data.groupby("Contract")["Churn"].mean() * 100

print("\n========================================")
print("       CHURN BY CONTRACT")
print("========================================")

print(contract_churn)


# ============================================================
# 7. AVERAGE SUPPORT CALLS BY CHURN
# ============================================================

support_analysis = data.groupby("Churn")["Support_Calls"].mean()

print("\n========================================")
print("       SUPPORT CALL ANALYSIS")
print("========================================")

print(support_analysis)

print(
    f"\nAverage support calls for customers who stayed: "
    f"{support_analysis.get(0, 0):.2f}"
)

print(
    f"Average support calls for customers who churned: "
    f"{support_analysis.get(1, 0):.2f}"
)


# ============================================================
# 8. AVERAGE MONTHLY CHARGES BY CHURN
# ============================================================

charge_analysis = data.groupby("Churn")["Monthly_Charges"].mean()

print("\n========================================")
print("       MONTHLY CHARGE ANALYSIS")
print("========================================")

print(charge_analysis)

print(
    f"\nAverage monthly charge for customers who stayed: "
    f"₹{charge_analysis.get(0, 0):.2f}"
)

print(
    f"Average monthly charge for customers who churned: "
    f"₹{charge_analysis.get(1, 0):.2f}"
)


# ============================================================
# 9. HIGH-RISK CUSTOMERS
# ============================================================

high_risk = data[
    (data["Support_Calls"] >= 5) &
    (data["Contract"] == "Monthly")
]

print("\n========================================")
print("       HIGH-RISK CUSTOMERS")
print("========================================")

print(
    high_risk[
        [
            "CustomerID",
            "Age",
            "Tenure_Months",
            "Monthly_Charges",
            "Support_Calls",
            "Contract",
            "Churn"
        ]
    ]
)


# ============================================================
# 10. VISUALIZATION — CHURN DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))

data["Churn"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status (0 = Stayed, 1 = Churned)")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "churn_distribution.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 11. VISUALIZATION — CONTRACT VS CHURN
# ============================================================

contract_churn.plot(
    kind="bar",
    figsize=(7, 5)
)

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "churn_by_contract.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 12. VISUALIZATION — SUPPORT CALLS VS CHURN
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    data["Support_Calls"],
    data["Monthly_Charges"],
    c=data["Churn"]
)

plt.title("Support Calls vs Monthly Charges")
plt.xlabel("Number of Support Calls")
plt.ylabel("Monthly Charges (₹)")

plt.tight_layout()

plt.savefig(
    "support_vs_charges.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 13. PREPARE DATA FOR MACHINE LEARNING
# ============================================================

print("\n========================================")
print("       MACHINE LEARNING")
print("========================================")

# Remove CustomerID because it does not help prediction

X = data.drop(
    columns=["CustomerID", "Churn"]
)

y = data["Churn"]


# Identify categorical columns

categorical_features = ["Contract"]

numeric_features = [
    "Age",
    "Tenure_Months",
    "Monthly_Charges",
    "Support_Calls"
]


# Convert categorical data into numerical format

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# ============================================================
# 14. SPLIT DATA INTO TRAINING AND TESTING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("Training customers:", len(X_train))
print("Testing customers:", len(X_test))


# ============================================================
# 15. CREATE MACHINE LEARNING MODEL
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)


# ============================================================
# 16. TRAIN THE MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)

print("\nModel training completed successfully.")


# ============================================================
# 17. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 18. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n========================================")
print("       MODEL RESULTS")
print("========================================")

print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 19. PREDICT NEW CUSTOMERS
# ============================================================

new_customers = pd.DataFrame({
    "Age": [23, 45, 30],
    "Tenure_Months": [4, 48, 10],
    "Monthly_Charges": [900, 1700, 1200],
    "Support_Calls": [6, 1, 4],
    "Contract": ["Monthly", "Yearly", "Monthly"]
})

predictions = model.predict(new_customers)

prediction_probabilities = model.predict_proba(
    new_customers
)[:, 1]


print("\n========================================")
print("       NEW CUSTOMER PREDICTIONS")
print("========================================")

for i in range(len(new_customers)):

    status = "Likely to Churn" if predictions[i] == 1 else "Likely to Stay"

    probability = prediction_probabilities[i] * 100

    print(
        f"Customer {i + 1}: {status} "
        f"(Churn Probability: {probability:.2f}%)"
    )


# ============================================================
# 20. SAVE PREDICTIONS
# ============================================================

new_customers["Predicted_Churn"] = predictions

new_customers["Churn_Probability"] = (
    prediction_probabilities * 100
).round(2)

new_customers.to_csv(
    "churn_predictions.csv",
    index=False
)

print("\nPredictions saved as churn_predictions.csv")


# ============================================================
# 21. SAVE CLEAN DATASET
# ============================================================

data.to_csv(
    "customers_cleaned.csv",
    index=False
)

print("Clean dataset saved as customers_cleaned.csv")


# ============================================================
# 22. FINAL INSIGHTS
# ============================================================

print("\n========================================")
print("           FINAL INSIGHTS")
print("========================================")

print(
    f"Total customers analyzed: {len(data)}"
)

print(
    f"Customers who churned: {churn_count.get(1, 0)}"
)

print(
    f"Customers who stayed: {churn_count.get(0, 0)}"
)

print(
    f"Overall churn rate: {churn_rate:.2f}%"
)

print(
    f"Highest contract churn rate: "
    f"{contract_churn.idxmax()}"
)

print(
    f"Highest contract churn percentage: "
    f"{contract_churn.max():.2f}%"
)

print(
    f"Machine learning model accuracy: "
    f"{accuracy * 100:.2f}%"
)

print("\n========================================")
print("       PROJECT COMPLETED")
print("========================================")