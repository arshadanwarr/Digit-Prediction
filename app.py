import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("mnist_model.h5")  # Ensure model.h5 is in the same directory

# Define class names (Update according to your dataset)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # Replace with actual class labels

st.title("Image Classification App")
st.title("Project by Arshad Anwar")
st.write("Upload an image, and the model will predict its class.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = image.resize((28, 28))  # Resize to match model input
    image_array = np.array(image) / 255.0  # Normalize

    # Reshape to (1, 180, 180, 1) for grayscale input
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension

    # Make a prediction
    prediction = model.predict(image_array)
    print("Prediction Output:", prediction)  # Debugging

    # Ensure the output shape matches class_names
    if prediction.shape[1] != len(class_names):
      st.error(f"Model output shape {prediction.shape} does not match class_names length {len(class_names)}")
    else:
      predicted_class = class_names[np.argmax(prediction)]
      st.write(f"**Prediction:** {predicted_class}")
      st.write(f"**Confidence:** {np.max(prediction) * 100:.2f}%")
