import google.generativeai as genai
import pandas as pd

# -----------------------------------
# Gemini API Key
# -----------------------------------

GEMINI_API_KEY = "AQ.Ab8RN6JJpvUWKUtWY0Ru06-T0N33dKousJQC3MG5G-XcW2EqXA"

genai.configure(
    api_key=GEMINI_API_KEY
)

model = genai.GenerativeModel(
    "gemini-3.5-flash-lite"
)

# -----------------------------------
# Load Top 100 Patients
# -----------------------------------

df = pd.read_csv(
    "reports/top100_patients.csv"
)

# -----------------------------------
# Generate Nurse Note
# -----------------------------------

def generate_nurse_note(patient):

    prompt = f"""
You are a healthcare nurse assistant.

Patient Risk Score:
{round(patient['Risk_Score'] * 100, 2)}%

Risk Level:
{patient['Risk_Level']}

Risk Factors:
{patient['Risk_Factors']}

Age:
{patient['Age']}

BMI:
{patient['BMI']}

Glucose:
{patient['Glucose']}

Generate:

1. Nurse Summary
2. Discussion Points
3. Follow-Up Recommendation

Start response with:

PENDING_APPROVAL
"""

    response = model.generate_content(prompt)

    return response.text


# -----------------------------------
# Generate Notes for Top 10 Patients
# -----------------------------------

top10 = df.head(10)

notes = []

for index, row in top10.iterrows():

    print(
        f"Generating note for Patient Rank {row['Rank']}..."
    )

    note = generate_nurse_note(row)

    notes.append(note)

# Add notes
top10["Nurse_Note"] = notes

# Save
top10.to_csv(
    "reports/top10_nurse_notes.csv",
    index=False
)

print(
    "\nNurse notes generated successfully!"
)

print(
    "Saved to reports/top10_nurse_notes.csv"
)