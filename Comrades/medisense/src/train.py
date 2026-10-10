import os
import pandas as pd
import numpy as np

from preprocessing import preprocess_data

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_validate

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import mlflow

# Create reports folder
os.makedirs("reports", exist_ok=True)

# -------------------------
# Load Data
# -------------------------

df = preprocess_data("data/diabetes.csv")

# Features & Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# -------------------------
# Cross Validation Setup
# -------------------------

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42
)

# -------------------------
# Models
# -------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

# -------------------------
# Metrics
# -------------------------

scoring = {
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

results = []

# -------------------------
# Training
# -------------------------

for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    mean_recall = scores["test_recall"].mean()
    mean_f1 = scores["test_f1"].mean()
    mean_auc = scores["test_roc_auc"].mean()

    print(f"Recall : {mean_recall:.4f}")
    print(f"F1     : {mean_f1:.4f}")
    print(f"AUC    : {mean_auc:.4f}")

    # MLflow Logging
    with mlflow.start_run(run_name=model_name):

        mlflow.log_metric("Recall", mean_recall)
        mlflow.log_metric("F1", mean_f1)
        mlflow.log_metric("ROC_AUC", mean_auc)

    # Save Results
    results.append({
        "Model": model_name,
        "Recall": round(mean_recall, 4),
        "F1": round(mean_f1, 4),
        "ROC_AUC": round(mean_auc, 4)
    })

# -------------------------
# Save Results
# -------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "reports/model_results.csv",
    index=False
)

print("\n====================")
print("MODEL COMPARISON")
print("====================")

print(results_df)

print("\nResults saved to:")
print("reports/model_results.csv")