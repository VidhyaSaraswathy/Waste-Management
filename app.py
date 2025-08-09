import streamlit as st
from PIL import Image
import numpy as np

st.title("Waste Management Classifier (Demo)")

uploaded_file = st.file_uploader("Upload an image of waste", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Dummy prediction logic: classify based on average color
    img_array = np.array(image.resize((50, 50)))  # resize for speed
    
    avg_color = img_array.mean(axis=(0, 1))
    
    # Simple rule: more green-ish = recyclable, else non-recyclable (just example)
    if avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:
        prediction = "Recyclable Waste ♻️"
    else:
        prediction = "Non-Recyclable Waste 🚮"
    
    st.write(f"**Prediction:** {prediction}")
