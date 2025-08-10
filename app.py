import streamlit as st
from PIL import Image
import numpy as np

# Dummy predict function without TensorFlow
def dummy_predict(img_array):
    # Simple dummy logic: if average pixel brightness > 0.5, predict recyclable
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
    img_array = img_array.astype(np.float32)

    prediction = dummy_predict(img_array)
    predicted_label = classes[prediction]

    st.write(f"**Predicted Waste Type:** {predicted_label}")
