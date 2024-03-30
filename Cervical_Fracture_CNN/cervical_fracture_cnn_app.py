import streamlit as st
import numpy as np
from PIL import Image
from PIL import ImageOps
from keras.models import load_model
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
model = load_model('model_fracture2.h5')
classes = ['FRACTURE','NORMAL']
def predict(image):
    img = Image.open(image)
    img = img.resize((150, 150))  # Resize image to match model input size
    img = ImageOps.grayscale(img)  # Convert image to grayscale
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    return classes[predicted_class], confidence

# Streamlit UI
st.title("CERVICAL FRACTURE PREDICTION")
img = Image.open("fracture_cervical.jpg")
st.image(img, width=600)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")


    # Make predictions
    # Display prediction results
    class_name, confidence = predict(uploaded_file)
    st.write(f"Prediction: {class_name}, Confidence: {confidence:.2f}")
