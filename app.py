import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('digit_recognition_model.h5')

st.title("Handwritten Digit Recognitions")

# Upload image
uploaded_image = st.file_uploader("Upload an image of a digit", type=["png", "jpg", "jpeg"])

def preprocess_image(image):
    # Convert to grayscale
    image = image.convert("L")
    # Resize to 28x28
    image = image.resize((28, 28))
    # Convert to numpy array
    image_array = np.array(image)
    # Normalize to the range [0, 1]
    image_array = image_array / 255.0
    # Flatten the image to match input shape (1, 784)
    image_array = image_array.reshape(1, 784)
    return image_array

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Predict digit
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction[0])

    st.write(f"Predicted Digit: {predicted_digit}")
