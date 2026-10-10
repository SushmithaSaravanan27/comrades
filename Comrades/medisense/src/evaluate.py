import os
import joblib
import pandas as pd

from preprocessing import preprocess_data

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# -------------------------
# Create models folder
# -------------------------

os.makedirs("models", exist_ok=True)

# -------------------------
# Read Model Results
# -------------------------

results = pd.read_csv("reports/model_results.csv")

print("\nModel Results:\n")
print(results)

# -------------------------
# Select Best Model
# -------------------------

best_model_name = results.sort_values(
    by="Recall",
    ascending=False
).iloc[0]["Model"]

print(f"\nBest Model Selected: {best_model_name}")

# -------------------------
# Load Entire Dataset
# -------------------------

df = preprocess_data("data/diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# -------------------------
# Recreate Winning Model
# -------------------------

if best_model_name == "Logistic Regression":

    model = LogisticRegression(
        max_iter=1000
    )

elif best_model_name == "Random Forest":

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    )

else:

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

# -------------------------
# Train Full Dataset
# -------------------------

print("\nTraining Best Model On Full Dataset...")

model.fit(X, y)

# -------------------------
# Save Model
# -------------------------

joblib.dump(
    model,
    "models/best_model.pkl"
)

print("\nModel Saved Successfully!")
print("Location: models/best_model.pkl")