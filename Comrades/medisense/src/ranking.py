import pandas as pd
import joblib

from preprocessing import preprocess_data

# -------------------------
# Load Data
# -------------------------

df = preprocess_data("data/diabetes.csv")

# Keep copy
output_df = df.copy()

# -------------------------
# Load Model
# -------------------------

model = joblib.load(
    "models/best_model.pkl"
)

# -------------------------
# Features
# -------------------------

X = df.drop(
    columns=["Outcome"]
)

# -------------------------
# Predict Risk Scores
# -------------------------

risk_scores = model.predict_proba(X)[:, 1]

output_df["Risk_Score"] = risk_scores

# -------------------------
# Risk Level
# -------------------------

def risk_level(score):

    if score >= 0.85:
        return "Critical"

    elif score >= 0.70:
        return "High"

    elif score >= 0.40:
        return "Medium"

    return "Low"


output_df["Risk_Level"] = output_df[
    "Risk_Score"
].apply(risk_level)

# -------------------------
# Risk Factors
# -------------------------

def get_risk_factors(row):

    factors = []

    if row["Glucose"] > 140:
        factors.append("High Glucose")

    if row["BMI"] > 30:
        factors.append("High BMI")

    if row["Age"] > 45:
        factors.append("Age > 45")

    if row["DiabetesPedigreeFunction"] > 0.8:
        factors.append("Family History")

    if not factors:
        factors.append("General Risk")

    return ", ".join(factors)


output_df["Risk_Factors"] = output_df.apply(
    get_risk_factors,
    axis=1
)

# -------------------------
# Rank Patients
# -------------------------

output_df = output_df.sort_values(
    by="Risk_Score",
    ascending=False
)

output_df["Rank"] = range(
    1,
    len(output_df) + 1
)

# -------------------------
# Top 100
# -------------------------

top100 = output_df.head(100)

# -------------------------
# Save CSV
# -------------------------

top100.to_csv(
    "reports/top100_patients.csv",
    index=False
)

print("\nTop 100 Patients Generated Successfully!")

print(
    "\nSaved To: reports/top100_patients.csv"
)

print(
    "\nTop 10 Patients:"
)

print(
    top100[
        [
            "Rank",
            "Risk_Score",
            "Risk_Level",
            "Risk_Factors"
        ]
    ].head(10)
)