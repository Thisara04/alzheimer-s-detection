import streamlit as st
import numpy as np
from PIL import Image

from model import build_model

# =====================
# App Configuration
# =====================
st.set_page_config(
    page_title="Alzheimer MRI Classifier",
    layout="centered"
)

st.title("🧠 Alzheimer’s MRI Image Classifier")
st.write(
    "Upload a **brain MRI image** to predict the Alzheimer’s stage.\n\n"
    "**⚠️ Educational use only — not a medical diagnostic tool.**"
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

ENTROPY_THRESHOLD = 1.20      # Primary rejection criterion
CONFIDENCE_THRESHOLD = 0.60   # Secondary safeguard

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
    img = np.array(image, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =====================
# Entropy Calculation
# =====================
def calculate_entropy(preds):
    return float(-np.sum(preds * np.log(preds + 1e-8)))

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

    # Preprocess & Predict
    input_tensor = preprocess_image(image)
    preds = model.predict(input_tensor)[0]

    confidence = float(np.max(preds))
    predicted_class = int(np.argmax(preds))
    entropy = calculate_entropy(preds)

    # =====================
    # Entropy-based Rejection
    # =====================
    if entropy > ENTROPY_THRESHOLD or confidence < CONFIDENCE_THRESHOLD:
        st.error(
            "⚠️ The uploaded image does **not appear to be a valid brain MRI**.\n\n"
            "Please upload a clear MRI scan of the brain."
        )

        st.caption(
            f"Entropy: {entropy:.2f} | Confidence: {confidence:.2f}"
        )

    else:
        st.success("✅ Prediction Successful")

        st.subheader("🧠 Prediction Result")
        st.write(
            f"**Predicted Stage:** {CLASS_NAMES[predicted_class]}\n\n"
            f"**Confidence:** {confidence * 100:.2f}%"
        )

        st.subheader("📊 Class Probabilities")
        for i, class_name in enumerate(CLASS_NAMES):
            st.write(f"{class_name}: {preds[i] * 100:.2f}%")

        st.caption(
            f"Entropy: {entropy:.2f} (lower = more confident)"
        )

# =====================
# Footer
# =====================
st.markdown("---")
st.caption(
    "This application is intended for educational and research purposes only. "
    "Always consult qualified medical professionals for diagnosis."
)
