import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# Load model and processor
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
        result_text = f"Predicted Class: {predicted_class}"
        # Increase font size using HTML tags
        st.markdown(f"<p style='font-size:20px'>{result_text}</p>", unsafe_allow_html=True)
