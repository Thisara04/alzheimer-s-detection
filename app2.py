import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

from model import build_model

# =====================
# App Config
# =====================
st.set_page_config(
    page_title="Alzheimer MRI Classifier",
    layout="centered"
)

st.title("🧠 Alzheimer’s MRI Image Classifier")
st.write(
    "Upload a **brain MRI image** to predict the Alzheimer’s stage.\n\n"
    "**⚠️ This tool is for educational purposes only.**"
)

# =====================
# Constants
# =====================
IMG_SIZE = 224
CLASS_NAMES = [
    "Mild Impairment",
    "Moderate Impairment",
    "No Impairment",
    "Very Mild Impairment"
]

CONFIDENCE_THRESHOLD = 0.60  # Reject non-MRI / uncertain images

# =====================
# Load Model (Cached)
# =====================
@st.cache_resource
def load_model():
    model = build_model()
    model.load_weights("alzheimers_mri_efficientnetB2.weights.h5")
    return model

model = load_model()

# =====================
# Image Preprocessing
# =====================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =====================
# File Upload
# =====================
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_tensor = preprocess_image(image)

    # Predict
    predictions = model.predict(input_tensor)
    confidence = float(np.max(predictions))
    predicted_class = int(np.argmax(predictions))

    # =====================
    # Safety Check
    # =====================
    if confidence < CONFIDENCE_THRESHOLD:
        st.error(
            "⚠️ The uploaded image does **not appear to be a valid brain MRI**.\n\n"
            "Please upload a clear MRI scan of the brain."
        )
    else:
        st.success("✅ Prediction Successful")

        st.subheader("🧠 Prediction Result")
        st.write(
            f"**Predicted Stage:** {CLASS_NAMES[predicted_class]}\n\n"
            f"**Confidence:** {confidence * 100:.2f}%"
        )

        # Probability breakdown
        st.subheader("📊 Class Probabilities")
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"{class_name}: {predictions[0][i] * 100:.2f}%")

# =====================
# Footer
# =====================
st.markdown("---")
st.caption(
    "This application is not a medical diagnostic tool. "
    "Consult a qualified medical professional for diagnosis."
)

