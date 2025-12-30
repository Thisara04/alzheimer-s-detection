import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image

# Load the model
try:
    from model import build_model
except ImportError:
    st.error("Please ensure 'model.py' is in the same directory.")

# ==========================================
# 1. Setup & Constants
# ==========================================
st.set_page_config(page_title="Alzheimer's Suite", layout="wide")

@st.cache_resource
def load_risk_model():
    return joblib.load("ensemble_calibrated.pkl")

@st.cache_resource
def load_mri_model():
    model = build_model()
    model.load_weights("alzheimers_mri_efficientnetB2.weights.h5")
    return model

THRESHOLD_RISK = 0.45
IMG_SIZE = 224
CLASS_NAMES = ["Mild Impairment", "Moderate Impairment", "No Impairment", "Very Mild Impairment"]
COUNTRY_LIST = ["Australia","Brazil","Canada","China","France","Germany","India","Italy","Japan","Mexico","Norway","Russia","Saudi Arabia","South Africa","South Korea","Spain","Sweden","UK","USA"]

# ==========================================
# 2. MRI Validation Logic
# ==========================================
def validate_is_mri(img):
    """Checks if the image matches MRI structural profiles vs digits/noise."""
    img_gray = img.convert("L").resize((100, 100))
    arr = np.array(img_gray)
    
    # 1. Edge/Corner Check: Real MRIs are usually black in corners
    corners = [arr[:10,:10], arr[:10,-10:], arr[-10:,:10], arr[-10:,-10:]]
    corner_avg = np.mean(corners)
    
    # 2. Intensity Variance: Digits/Text have high contrast (extreme black/white)
    # MRIs have a smoother grayscale distribution (organic)
    std_dev = np.std(arr)
    
    if corner_avg > 60: # Too bright for a medical scan background
        return False, "Image background is too bright. Medical MRIs usually have dark backgrounds."
    if std_dev > 90: # Too much high-contrast (likely text or a digit)
        return False, "High contrast detected. This looks like a graphic or digit, not an MRI."
    return True, "Success"

# ==========================================
# 3. Sidebar & Navigation
# ==========================================
st.sidebar.title("🧠 Alzheimer's Suite")
mode = st.sidebar.radio("Module", ["Risk Assessment", "MRI Classifier"])

# ==========================================
# 4. Lifestyle Module (Using your exact Mapping)
# ==========================================
if mode == "Risk Assessment":
    st.title("📋 Clinical Risk Calculator")
    
    with st.form("input_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 40, 100, 65)
            gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
            edu_map = {"No":0, "Primary":5, "Secondary":8, "High":12, "Diploma":13, "Bachelor":16, "Master":18, "PhD":21}
            edu = st.selectbox("Education Level", list(edu_map.keys()))
            bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
            physical = st.selectbox("Physical Activity", [0,1,2], format_func=lambda x: ["Low","Medium","High"][x])
            smoking = st.selectbox("Smoking Status", [0,1,2], format_func=lambda x: ["Never","Former","Current"][x])
            alcohol = st.selectbox("Alcohol Consumption", [0,1,2], format_func=lambda x: ["Never","Occasionally","Regularly"][x])
            diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
            hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
            cholesterol = st.selectbox("Cholesterol", [0, 1], format_func=lambda x: "Normal" if x==0 else "High")
            family = st.selectbox("Family History", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        
        with c2:
            cog_score = st.slider("Cognitive Score", 0, 100, 70)
            depression = st.selectbox("Depression", [0,1,2], format_func=lambda x: ["Low","Medium","High"][x])
            sleep = st.selectbox("Sleep Quality", [0,1,2], format_func=lambda x: ["Poor","Average","Good"][x])
            diet = st.selectbox("Dietary Habits", [0,1,2], format_func=lambda x: ["Healthy","Average","Unhealthy"][x])
            pollution = st.selectbox("Air Pollution Exposure", [0,1,2], format_func=lambda x: ["Low","Medium","High"][x])
            employment = st.selectbox("Employment", [0,1,2], format_func=lambda x: ["Employed","Unemployed","Retired"][x])
            marital = st.selectbox("Marital Status", [0,1,2], format_func=lambda x: ["Single","Widowed","Married"][x])
            genetic = st.selectbox("Genetic Risk Factor(APOE-ε4)", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
            social = st.selectbox("Social Engagement", [0,1,2], format_func=lambda x: ["Low","Medium","High"][x])
            income = st.selectbox("Income (compared to the country you live in)", [0,1,2], format_func=lambda x: ["Low","Medium","High"][x])
            stress = st.selectbox("Stress Level", [0,1,2], format_func=lambda x: ["Low","Medium","High"][x])
            living = st.selectbox("Living Area", [0, 1], format_func=lambda x: "Urban" if x==0 else "Rural")
        
        country = st.selectbox("Country(select the most relevant or related country to you)", COUNTRY_LIST)
        submit = st.form_submit_button("Predict Risk")

    if submit:
        # Construct exact column order based on your list
        row = {
            'Age': age, 'Gender': gender, 'Education Level': edu_map[edu], 'BMI': bmi,
            'Physical Activity Level': physical, 'Smoking Status': smoking, 'Alcohol Consumption': alcohol,
            'Diabetes': diabetes, 'Hypertension': hypertension, 'Cholesterol Level': cholesterol,
            'Family History of Alzheimer’s': family, 'Cognitive Test Score': cog_score,
            'Depression Level': depression, 'Sleep Quality': sleep, 'Dietary Habits': diet,
            'Air Pollution Exposure': pollution, 'Employment Status': employment, 'Marital Status': marital,
            'Genetic Risk Factor (APOE-ε4 allele)': genetic, 'Social Engagement Level': social,
            'Income Level': income, 'Stress Levels': stress, 'Urban vs Rural Living': living
        }
        for c in COUNTRY_LIST:
            row[f"Country_{c}"] = 1 if country == c else 0
        
        df_input = pd.DataFrame([row])
        risk_model = load_risk_model()
        proba = risk_model.predict_proba(df_input)[0][1]
        
        st.metric("Probability", f"{proba:.2%}")
        if proba > THRESHOLD_RISK: st.error("Higher Risk Detected")
        else: st.success("Lower Risk Detected")

# ==========================================
# 5. MRI Module (With Digit/Noise Filtering)
# ==========================================
else:
    st.title("🧠 MRI Stage Classifier")
    uploaded_file = st.file_uploader("Upload Scan", type=["jpg","png","jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, width=300)
        
        # VALIDATION GATE
        is_mri, msg = validate_is_mri(img)
        
        if not is_mri:
            st.error(f"❌ Rejected: {msg}")
        else:
            mri_model = load_mri_model()
            # Preprocess
            img_prep = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            img_arr = np.array(img_prep) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)
            
            preds = mri_model.predict(img_arr)[0]
            conf = np.max(preds)
            
            if conf < 0.60:
                st.warning("⚠️ Unclear Image. This might not be a standard Brain MRI.")
            else:
                st.success(f"Result: {CLASS_NAMES[np.argmax(preds)]} ({conf:.2%})")
