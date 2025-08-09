app_code = 
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("waste_classifier.h5")
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

st.title("♻️ Waste Classification App")
st.write("Upload an image to classify waste type")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((224,224))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    label = classes[np.argmax(prediction)]

    st.write(f"**Predicted Waste Type:** {label}")


with open("app.py", "w") as f:
    f.write(app_code)

print("Streamlit app script 'app.py' created.")
