import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json

# Load trained model
model = tf.keras.models.load_model("leaf_disease_model.h5")

# Load disease info from JSON
with open("disease_info.json", "r") as file:
    disease_info = json.load(file)

# Get class labels from JSON keys
class_labels = list(disease_info.keys())

st.title("🌿 Plant Disease Detector")
st.write("Upload a plant leaf image to detect the disease and get a cure suggestion.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Leaf", use_column_width=True)

    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]

    # Get info from JSON
    info = disease_info[predicted_class]
    st.subheader(f"Plant Type: {info['plant']}")
    st.subheader(f"Disease: {info['disease']}")
    st.subheader(f"Cure: {info['cure']}")
