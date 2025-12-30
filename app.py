import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Load calibrated ensemble model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("ensemble_calibrated.pkl")

model = load_model()
THRESHOLD = 0.45

st.set_page_config(page_title="Alzheimer’s Risk Prediction", layout="centered")
st.title("🧠 Alzheimer’s Disease Risk Prediction")
st.caption("This tool estimates risk probability — not a medical diagnosis.")
st.divider()

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Patient Information")

age = st.slider("Age", 40, 100, 65)

gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

# ✅ Education level (interpretable → numeric)
education_label = st.selectbox(
    "Highest Education Level",
    [
        "No formal education",
        "Primary school",
        "Secondary school",
        "High school",
        "Diploma / Vocational",
        "Bachelor’s degree",
        "Master’s degree",
        "Doctorate / PhD"
    ]
)

education_map = {
    "No formal education": 0,
    "Primary school": 5,
    "Secondary school": 8,
    "High school": 12,
    "Diploma / Vocational": 13,
    "Bachelor’s degree": 16,
    "Master’s degree": 18,
    "Doctorate / PhD": 21
}
education = education_map[education_label]

bmi = st.slider("BMI", 15.0, 40.0, 25.0)

physical = st.selectbox(
    "Physical Activity Level", [0, 1, 2],
    format_func=lambda x: ["Low", "Medium", "High"][x]
)

smoking = st.selectbox(
    "Smoking Status", [0, 1, 2],
    format_func=lambda x: ["Never", "Former", "Current"][x]
)

alcohol = st.selectbox(
    "Alcohol Consumption", [0, 1, 2],
    format_func=lambda x: ["Never", "Occasionally", "Regularly"][x]
)

diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
cholesterol = st.selectbox("Cholesterol Level", [0, 1], format_func=lambda x: "Normal" if x == 0 else "High")
family_history = st.selectbox("Family History of Alzheimer’s", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

cognitive_score = st.slider("Cognitive Test Score", 0, 100, 70)

depression = st.selectbox(
    "Depression Level", [0, 1, 2],
    format_func=lambda x: ["Low", "Medium", "High"][x]
)

sleep = st.selectbox(
    "Sleep Quality", [0, 1, 2],
    format_func=lambda x: ["Poor", "Average", "Good"][x]
)

diet = st.selectbox(
    "Dietary Habits", [0, 1, 2],
    format_func=lambda x: ["Healthy", "Average", "Unhealthy"][x]
)

pollution = st.selectbox(
    "Air Pollution Exposure", [0, 1, 2],
    format_func=lambda x: ["Low", "Medium", "High"][x]
)

employment = st.selectbox(
    "Employment Status", [0, 1, 2],
    format_func=lambda x: ["Employed", "Unemployed", "Retired"][x]
)

marital = st.selectbox(
    "Marital Status", [0, 1, 2],
    format_func=lambda x: ["Single", "Widowed", "Married"][x]
)

genetic = st.selectbox(
    "Genetic Risk Factor (APOE-ε4 allele)", [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

social = st.selectbox(
    "Social Engagement Level", [0, 1, 2],
    format_func=lambda x: ["Low", "Medium", "High"][x]
)

income = st.selectbox(
    "Income Level", [0, 1, 2],
    format_func=lambda x: ["Low", "Medium", "High"][x]
)

stress = st.selectbox(
    "Stress Levels", [0, 1, 2],
    format_func=lambda x: ["Low", "Medium", "High"][x]
)

living = st.selectbox(
    "Urban vs Rural Living", [0, 1],
    format_func=lambda x: "Urban" if x == 0 else "Rural"
)

# -----------------------------
# Country (One-Hot Encoding)
# -----------------------------
country_list = [
    "Australia","Brazil","Canada","China","France","Germany","India","Italy",
    "Japan","Mexico","Norway","Russia","Saudi Arabia","South Africa",
    "South Korea","Spain","Sweden","UK","USA"
]

country = st.selectbox("Country", country_list)

# -----------------------------
# Encode input
# -----------------------------
def encode_input():
    data = {
        "Age": age,
        "Gender": gender,
        "Education Level": education,
        "BMI": bmi,
        "Physical Activity Level": physical,
        "Smoking Status": smoking,
        "Alcohol Consumption": alcohol,
        "Diabetes": diabetes,
        "Hypertension": hypertension,
        "Cholesterol Level": cholesterol,
        "Family History of Alzheimer’s": family_history,
        "Cognitive Test Score": cognitive_score,
        "Depression Level": depression,
        "Sleep Quality": sleep,
        "Dietary Habits": diet,
        "Air Pollution Exposure": pollution,
        "Employment Status": employment,
        "Marital Status": marital,
        "Genetic Risk Factor (APOE-ε4 allele)": genetic,
        "Social Engagement Level": social,
        "Income Level": income,
        "Stress Levels": stress,
        "Urban vs Rural Living": living
    }

    # One-hot countries
    for c in country_list:
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

    st.metric("Predicted Probability", f"{proba:.2%}")

    if prediction == 1:
        st.error("⚠️ Higher Risk of Alzheimer’s Disease")
    else:
        st.success("✅ Lower Risk of Alzheimer’s Disease")

    st.caption(f"Decision threshold: {THRESHOLD}")
