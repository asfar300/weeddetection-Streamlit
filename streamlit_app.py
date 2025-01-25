import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from tensorflow.keras.models import load_model

# Streamlit title
st.title("Weed Paddy Detection")

# Upload image section
uploaded_image = st.file_uploader("Upload a weed image", type=["jpg", "png", "jpeg"])

# Load your model (adjust according to your model type)
@st.cache_resource
def load_trained_model():


    # If using TensorFlow/Keras:
     model = load_model('C:\Users\Asfar\Downloads\Weed_detection_using_CNN (2).h5')
     return model

model = load_trained_model()

if uploaded_image is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image (ensure it matches your model's input requirements)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust to your model's input size
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Model prediction
    with torch.no_grad():
        output = model(image_tensor)  # Modify this line based on your model's input-output

    # Display prediction
    predicted_class = output.argmax(dim=1).item()  # Modify based on your model output
    st.write(f"Predicted Class: {predicted_class}")
