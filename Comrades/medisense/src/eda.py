import os
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import preprocess_data

# Create reports directory if not exists
os.makedirs("reports", exist_ok=True)

# Load preprocessed data
df = preprocess_data("data/diabetes.csv")

print("\nStarting EDA...")

# ==========================
# 1. Outcome Distribution
# ==========================

plt.figure(figsize=(6, 4))

sns.countplot(
    x="Outcome",
    data=df,
    hue="Outcome",
    legend=False,
    palette="Set2"
)

plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("reports/outcome_distribution.png")
plt.close()

# ==========================
# 2. Correlation Heatmap
# ==========================

plt.figure(figsize=(12, 8))

corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("reports/correlation_heatmap.png")
plt.close()

# ==========================
# 3. Glucose vs Outcome
# ==========================

plt.figure(figsize=(6, 4))

sns.boxplot(
    x="Outcome",
    y="Glucose",
    data=df
)

plt.title("Glucose vs Outcome")

plt.tight_layout()
plt.savefig("reports/glucose_vs_outcome.png")
plt.close()

# ==========================
# 4. BMI vs Outcome
# ==========================

plt.figure(figsize=(6, 4))

sns.boxplot(
    x="Outcome",
    y="BMI",
    data=df
)

plt.title("BMI vs Outcome")

plt.tight_layout()
plt.savefig("reports/bmi_vs_outcome.png")
plt.close()

# ==========================
# 5. Age vs Outcome
# ==========================

plt.figure(figsize=(6, 4))

sns.boxplot(
    x="Outcome",
    y="Age",
    data=df
)

plt.title("Age vs Outcome")

plt.tight_layout()
plt.savefig("reports/age_vs_outcome.png")
plt.close()

# ==========================
# Correlation with Outcome
# ==========================

print("\nFeature Correlation With Outcome:\n")

print(
    corr["Outcome"]
    .sort_values(ascending=False)
)

print("\nEDA completed successfully!")
print("\nCharts saved under reports/")