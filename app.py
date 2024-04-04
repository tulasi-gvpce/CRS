import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests

# Load model and processor
# Load model directly
map={
    0:"No_Diabetice",
    1:"Mild",
    2:"Moderate",
    3:"severe",
    4:"proliferate_DR"
}
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("Kontawat/vit-diabetic-retinopathy-classification")
model = AutoModelForImageClassification.from_pretrained("Kontawat/vit-diabetic-retinopathy-classification")
# Function to make prediction
def predict(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax().item()
    return predicted_class

# Streamlit app
st.title("Diabetic Retinopathy Classification")
st.write("Upload an image for classification.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make prediction when button is clicked
    if st.button("Classify"):
        st.write("Classifying...")
        predicted_class = predict(image)
        st.write("Predicted Class:", map[predicted_class])
