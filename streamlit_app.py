import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Streamlit title
st.title("Weed Paddy Detection")

# Upload image section
uploaded_image = st.file_uploader("Upload a weed image", type=["jpg", "png", "jpeg"])

# Load your trained model (Keras)
@st.cache_resource
def load_trained_model():
    # Load the model (adjust path to your model file)
    model = load_model(‪C:\Users\Asfar\Downloads\Weed_detection_using_CNN (2).h)  # Update this with your model path
    return model

model = load_trained_model()

if uploaded_image is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image (ensure it matches your model's input size)
    image_array = np.array(image.resize((224, 224)))  # Resize to match model input size
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # If your model is expecting normalized values (like in most CNN models):
    image_array = image_array / 255.0  # Normalize pixel values to [0, 1]

    # Model prediction
    prediction = model.predict(image_array)
    
    # If your model has two classes, use argmax to get the predicted class
    predicted_class = np.argmax(prediction, axis=1)
    
    # Display prediction
    st.write(f"Predicted Class: {predicted_class[0]}")


