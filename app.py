import streamlit as st
from PIL import Image
import numpy as np
import pickle


# Load your pre-trained classical ML model (SVM, RandomForest, etc.)
model = pickle.load(open("model.pkl", "rb"))

# Define class labels
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

st.title("♻️ Waste Classification App (No TensorFlow)")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def extract_features(img):
    # Convert PIL image to OpenCV format
    img = np.array(img)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    # Resize for consistency
    img = cv2.resize(img, (100, 100))
    # Calculate color histogram for each channel
    features = []
    for i in range(3):  # RGB channels
        hist = cv2.calcHist([img], [i], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return np.array(features)

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Extract features and predict
    features = extract_features(img).reshape(1, -1)
    prediction = model.predict(features)
    label = classes[prediction[0]]
    
    st.write(f"**Predicted Waste Type:** {label}")

