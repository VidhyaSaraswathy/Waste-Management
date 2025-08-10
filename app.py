import streamlit as st
from PIL import Image
import numpy as np

def dummy_predict(img_array):
    brightness = np.mean(img_array)
    if brightness > 0.5:
        return 1  # Recyclable
    else:
        return 0  # Non-Recyclable

classes = ['Non-Recyclable', 'Recyclable']

st.title("♻️ Waste Classification App (No TensorFlow)")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(img) / 255.0

    prediction = dummy_predict(img_array)
    st.write(f"**Predicted Waste Type:** {classes[prediction]}")
