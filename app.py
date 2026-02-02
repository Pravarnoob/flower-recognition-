import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="Flower Recognition using CNN",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Flower Recognition using CNN")
st.write("Upload a flower image and get the predicted flower category.")

# -----------------------------
# Load SavedModel
# -----------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.keras")
    return model

model = load_model()

# -----------------------------
# UPDATE CLASS NAMES
# -----------------------------
class_names = [
    "Daisy",
    "Dandelion",
    "Rose",
    "Sunflower",
    "Tulip"
]

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))  # change if trained with different size
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a flower image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            img = preprocess_image(image)
            predictions = model.predict(img)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

        st.success(f"🌼 Predicted Flower: **{predicted_class}**")
        st.info(f"🔍 Confidence: **{confidence:.2f}%**")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("📘 **Academic Minor Project – Flower Recognition using CNN**")
