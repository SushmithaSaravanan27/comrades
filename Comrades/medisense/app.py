import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="MediSense",
    page_icon="🏥",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")


model = load_model()

# ==========================================
# LOAD TOP 100
# ==========================================

@st.cache_data
def load_top100():
    return pd.read_csv("reports/top100_patients.csv")


# ==========================================
# HEADER
# ==========================================

st.title("🏥 MediSense")
st.subheader("AI-Powered Diabetes Risk Prioritization System")

# ==========================================
# SIDEBAR
# ==========================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Top 100 Patients",
        "Patient Predictor",
        "AI Nurse Assistant"
    ]
)

# ==========================================
# DASHBOARD
# ==========================================

if page == "Dashboard":

    st.header("📊 Dashboard")

    try:
        df = pd.read_csv("data/diabetes.csv")

        total_patients = len(df)
        diabetes_rate = round(
            (df["Outcome"].mean() * 100),
            2
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Total Patients",
                total_patients
            )

        with col2:
            st.metric(
                "Diabetes Prevalence %",
                diabetes_rate
            )

        st.success(
            "Top-100 callback list generated successfully."
        )

    except Exception as e:
        st.error(str(e))

# ==========================================
# TOP 100 CALLBACK LIST
# ==========================================

elif page == "Top 100 Patients":

    st.header("📞 Top-100 Callback List")

    try:

        top100 = load_top100()

        st.dataframe(
            top100,
            use_container_width=True
        )

        csv = top100.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇ Download Top 100 CSV",
            csv,
            "top100_patients.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(
            f"Unable to load top100_patients.csv\n{e}"
        )

# ==========================================
# PATIENT PREDICTOR
# ==========================================

elif page == "Patient Predictor":

    st.header("🩺 Diabetes Risk Predictor")

    col1, col2 = st.columns(2)

    with col1:

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            value=1
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0,
            value=120
        )

        bp = st.number_input(
            "Blood Pressure",
            min_value=0,
            value=70
        )

        skin = st.number_input(
            "Skin Thickness",
            min_value=0,
            value=20
        )

    with col2:

        insulin = st.number_input(
            "Insulin",
            min_value=0,
            value=80
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            value=28.5
        )

        pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            value=0.5
        )

        age = st.number_input(
            "Age",
            min_value=1,
            value=35
        )

    if st.button("Predict Risk"):

        patient = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age,

            "Glucose_Missing": 1 if glucose == 0 else 0,
            "BloodPressure_Missing": 1 if bp == 0 else 0,
            "SkinThickness_Missing": 1 if skin == 0 else 0,
            "Insulin_Missing": 1 if insulin == 0 else 0,
            "BMI_Missing": 1 if bmi == 0 else 0
        }])

        risk_score = model.predict_proba(
            patient
        )[0][1]

        st.metric(
            "Risk Score",
            f"{risk_score*100:.2f}%"
        )

        if risk_score >= 0.85:
            st.error("🔴 Critical Risk")

        elif risk_score >= 0.70:
            st.warning("🟠 High Risk")

        elif risk_score >= 0.40:
            st.info("🟡 Medium Risk")

        else:
            st.success("🟢 Low Risk")

# ==========================================
# GEMINI NURSE ASSISTANT
# ==========================================

elif page == "AI Nurse Assistant":

    st.header("🤖 Gemini Nurse Assistant")

    api_key = st.text_input(
        "Gemini API Key",
        type="password"
    )

    risk_score = st.number_input(
        "Risk Score (%)",
        min_value=0.0,
        max_value=100.0,
        value=90.0
    )

    risk_factors = st.text_area(
        "Risk Factors",
        value="High Glucose, High BMI, Age > 45"
    )

    if st.button("Generate Nurse Note"):

        try:

            genai.configure(
                api_key=api_key
            )

            gemini_model = genai.GenerativeModel(
                "gemini-1.5-flash"
            )

            prompt = f"""
You are a healthcare nurse follow-up assistant.

Patient Risk Score:
{risk_score}%

Risk Factors:
{risk_factors}

Generate:

1. Nurse Summary
2. Discussion Points
3. Recommended Follow-Up

Start the response with:

PENDING_APPROVAL
"""

            response = gemini_model.generate_content(
                prompt
            )

            st.success("Nurse Note Generated Successfully")

            st.write(response.text)

        except Exception as e:

            st.error(
                f"Gemini error:\n{e}"
            )