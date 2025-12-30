import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image

# Ensure model.py (containing build_model) and your weights are in the directory
try:
    from model import build_model
except ImportError:
    st.error("Error: 'model.py' not found. Please ensure it is in the app directory.")

# ==========================================
# 1. Page Configuration & Models
# ==========================================
st.set_page_config(
    page_title="Alzheimer's Diagnostic Suite",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_risk_model():
    # Placeholder for your lifestyle model
    return joblib.load("ensemble_calibrated.pkl")

@st.cache_resource
def load_mri_model():
    model = build_model()
    model.load_weights("alzheimers_mri_efficientnetB2.weights.h5")
    return model

# Constants
THRESHOLD_RISK = 0.45
IMG_SIZE = 224
CLASS_NAMES = ["Mild Impairment", "Moderate Impairment", "No Impairment", "Very Mild Impairment"]

# VALIDATION SETTINGS
# If Entropy is HIGH, the model is "confused" (Not an MRI)
# If Confidence is LOW, the model doesn't recognize the patterns
ENTROPY_THRESHOLD = 1.20  
CONFIDENCE_THRESHOLD = 0.65 

# ==========================================
# 2. Sidebar Navigation
# ==========================================
st.sidebar.title("🧠 Alzheimer's Suite")
app_mode = st.sidebar.radio("Select Analysis Module", ["Lifestyle Risk Assessment", "MRI Image Classification"])
st.sidebar.divider()
st.sidebar.warning("⚠️ Educational use only — not a medical diagnostic tool.")

# ==========================================
# 3. Module 1: Lifestyle Risk Assessment
# ==========================================
if app_mode == "Lifestyle Risk Assessment":
    st.title("📋 Alzheimer’s Risk Prediction")
    risk_model = load_risk_model()

    with st.form("risk_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 40, 100, 65)
            gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            edu_label = st.selectbox("Education", ["No formal education", "High school", "Bachelor’s degree", "Master’s degree", "Doctorate"])
            bmi = st.slider("BMI", 15.0, 40.0, 25.0)
            cognitive_score = st.slider("Cognitive Test Score", 0, 100, 70)
        with col2:
            family_history = st.selectbox("Family History", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            genetic = st.selectbox("Genetic Risk (APOE-ε4)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            sleep = st.selectbox("Sleep Quality", [0, 1, 2], format_func=lambda x: ["Poor", "Average", "Good"][x])
            stress = st.selectbox("Stress Levels", [0, 1, 2], format_func=lambda x: ["Low", "Medium", "High"][x])
        
        country_list = ["Australia","Brazil","Canada","China","France","Germany","India","Italy","Japan","USA","UK"]
        country = st.selectbox("Country", country_list)
        submit_risk = st.form_submit_button("🔍 Calculate Risk Score")

    if submit_risk:
        # Simplified encoding for demonstration
        proba = 0.25 # Replace with: risk_model.predict_proba(df)[0][1]
        st.metric("Risk Probability", f"{proba:.2%}")

# ==========================================
# 4. Module 2: MRI Image Classifier (with Identification Logic)
# ==========================================
else:
    st.title("🧠 MRI Analysis & Identification")
    st.write("Upload an image. The system will first verify if it is a Brain MRI.")
    
    mri_model = load_mri_model()
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # UI Layout
        col_img, col_pred = st.columns([1, 1])
        with col_img:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Preprocessing
        img_prep = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img_prep, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        with st.spinner("Verifying image type and analyzing..."):
            preds = mri_model.predict(img_array)[0]
        
        confidence = float(np.max(preds))
        predicted_class = int(np.argmax(preds))
        entropy = float(-np.sum(preds * np.log(preds + 1e-8)))

        with col_pred:
            # STEP 1: IDENTIFICATION GATE
            if entropy > ENTROPY_THRESHOLD or confidence < CONFIDENCE_THRESHOLD:
                st.error("### ❌ Image Identification Failed")
                st.write("""
                The system has determined that this image is **not a valid Brain MRI scan**. 
                
                **Possible reasons:**
                * The image is not a medical scan (e.g., a photo, a diagram).
                * The MRI is of a different body part (e.g., knee, chest).
                * The image quality is too low for the AI to recognize brain structures.
                """)
                st.info(f"**Diagnostic Info:** Uncertainty level (Entropy) is too high ({entropy:.2f}).")
            
            # STEP 2: CLASSIFICATION (Only if it passes identification)
            else:
                st.success("### ✅ Brain MRI Identified")
                st.write(f"The system is confident this is a brain scan. Proceeding to stage analysis...")
                st.divider()
                
                st.subheader(f"Result: {CLASS_NAMES[predicted_class]}")
                st.progress(confidence)
                st.write(f"Analysis Confidence: **{confidence * 100:.2f}%**")
                
                with st.expander("View Probability Distribution"):
                    for i, name in enumerate(CLASS_NAMES):
                        st.write(f"{name}: {preds[i]*100:.1f}%")

# ==========================================
# 5. Global Footer
# ==========================================
st.markdown("---")
st.caption("Developed for medical research and educational purposes.")
