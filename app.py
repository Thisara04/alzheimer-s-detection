import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -----------------------------
# Load trained calibrated model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("ensemble_calibrated.pkl")

model = load_model()

THRESHOLD = 0.45

st.set_page_config(
    page_title="Alzheimer’s Risk Prediction",
    layout="centered"
)

st.title("🧠 Alzheimer’s Disease Risk Prediction")
st.markdown("Predict the probability of Alzheimer's diagnosis using patient data.")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Patient Information")

age = st.slider("Age", 40, 100, 65)
gender = st.selectbox("Gender", ["Female", "Male"])
education = st.slider("Education Level", 0, 20, 10)
bmi = st.slider("BMI", 15, 40, 25)
physical = st.selectbox("Physical Activity Level", ["Low", "Medium", "High"])
smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
alcohol = st.selectbox("Alcohol Consumption", ["Never", "Occasionally", "Regularly"])
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
cholesterol = st.selectbox("Cholesterol Level", ["Normal", "High"])
family_history = st.selectbox("Family History of Alzheimer’s", ["No", "Yes"])
cognitive_score = st.slider("Cognitive Test Score", 0, 100, 70)
depression = st.selectbox("Depression Level", ["Low", "Medium", "High"])
sleep = st.selectbox("Sleep Quality", ["Poor", "Average", "Good"])
diet = st.selectbox("Dietary Habits", ["Healthy", "Average", "Unhealthy"])
pollution = st.selectbox("Air Pollution Exposure", ["Low", "Medium", "High"])
employment = st.selectbox("Employment Status", ["Employed", "Unemployed", "Retired"])
marital = st.selectbox("Marital Status", ["Single", "Widowed", "Married"])
genetic = st.selectbox("Genetic Risk Factor (APOE-ε4)", ["No", "Yes"])
social = st.selectbox("Social Engagement Level", ["Low", "Medium", "High"])
income = st.selectbox("Income Level", ["Low", "Medium", "High"])
stress = st.selectbox("Stress Levels", ["Low", "Medium", "High"])
living = st.selectbox("Urban vs Rural Living", ["Urban", "Rural"])

country = st.selectbox("Country", [
    "Australia","Brazil","Canada","China","France","Germany","India","Italy",
    "Japan","Mexico","Norway","Russia","Saudi Arabia","South Africa",
    "South Korea","Spain","Sweden","UK","USA"
])

# -----------------------------
# Encode inputs
# -----------------------------
def encode_input():
    data = {
        "Age": age,
        "Gender": 1 if gender == "Male" else 0,
        "Education Level": education,
        "BMI": bmi,
        "Physical Activity Level": {"Low":0,"Medium":1,"High":2}[physical],
        "Smoking Status": {"Never":0,"Former":1,"Current":2}[smoking],
        "Alcohol Consumption": {"Never":0,"Occasionally":1,"Regularly":2}[alcohol],
        "Diabetes": 1 if diabetes == "Yes" else 0,
        "Hypertension": 1 if hypertension == "Yes" else 0,
        "Cholesterol Level": 1 if cholesterol == "High" else 0,
        "Family History of Alzheimer’s": 1 if family_history == "Yes" else 0,
        "Cognitive Test Score": cognitive_score,
        "Depression Level": {"Low":0,"Medium":1,"High":2}[depression],
        "Sleep Quality": {"Poor":0,"Average":1,"Good":2}[sleep],
        "Dietary Habits": {"Healthy":0,"Average":1,"Unhealthy":2}[diet],
        "Air Pollution Exposure": {"Low":0,"Medium":1,"High":2}[pollution],
        "Employment Status": {"Employed":0,"Unemployed":1,"Retired":2}[employment],
        "Marital Status": {"Single":0,"Widowed":1,"Married":2}[marital],
        "Genetic Risk Factor (APOE-ε4 allele)": 1 if genetic == "Yes" else 0,
        "Social Engagement Level": {"Low":0,"Medium":1,"High":2}[social],
        "Income Level": {"Low":0,"Medium":1,"High":2}[income],
        "Stress Levels": {"Low":0,"Medium":1,"High":2}[stress],
        "Urban vs Rural Living": 1 if living == "Rural" else 0,
    }

    # One-hot countries
    for c in [
        "Australia","Brazil","Canada","China","France","Germany","India","Italy",
        "Japan","Mexico","Norway","Russia","Saudi Arabia","South Africa",
        "South Korea","Spain","Sweden","UK","USA"
    ]:
        data[f"Country_{c}"] = 1 if country == c else 0

    return pd.DataFrame([data])

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Alzheimer’s Risk"):
    input_df = encode_input()

    proba = model.predict_proba(input_df)[0][1]
    prediction = int(proba >= THRESHOLD)

    st.divider()
    st.subheader("Prediction Result")

    st.metric("Alzheimer’s Probability", f"{proba:.2%}")

    if prediction == 1:
        st.error("⚠️ High Risk of Alzheimer’s Disease")
    else:
        st.success("✅ Low Risk of Alzheimer’s Disease")

    st.caption(f"Decision threshold: {THRESHOLD}")
