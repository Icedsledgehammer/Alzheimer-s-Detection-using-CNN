import numpy as np
from tensorflow.keras.models import load_model, save_model
from PIL import Image
import streamlit as st

model1 = load_model('alzheimer_model.h5')
input_shape = model1.layers[0].input_shape

# Define class labels
class_labels = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']

st.title("Image Classification Web App")

# Upload an image for classification
uploaded_image = st.file_uploader("Upload an image (Format allowed- jpg, jpeg, png)", type = ["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption = "Uploaded Image", use_column_width = True)

    # Preprocess and predict
    # Resize the image to match the model's input shape (176x208)
    image = image.resize((, 176))  # Swapped width and height to match your target shape

    # Convert the image to a NumPy array and normalize it
    image = np.array(image) / 255.0

    # Access the input shape of your model
    input_shape = model1.layers[0].input_shape
    
    if model1.input_shape[-1] == 3:
        image = image.reshape((1, 176, 208, 3))
    else:
        image = image.reshape((1, 176, 208, 1))

    result = model1.predict(image)
    class_id = np.argmax(result)
    class_name = class_labels[class_id]

    st.write(f"Prediction: {class_name}")
else:
    st.write("Please upload an image for classification.")
