import streamlit as st
from PIL import Image
import torch
import pickle
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer  # Import the necessary classes from transformers

# Load your feature extractor and tokenizer from .pkl files
def load_objects():
    with open('feature_extractor.pkl', 'rb') as feature_file:
        feature_extractor = pickle.load(feature_file)

    with open('tokenizer.pkl', 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    return feature_extractor, tokenizer

# Load your model here
def load_model():
    with open('model.pkl', 'rb') as model_file:  # Correctly load the pickled model
        model = pickle.load(model_file)
    return model

# Function to predict captions
def predict_step(uploaded_files, model, feature_extractor, tokenizer, device, gen_kwargs):
    images = []
    
    for uploaded_file in uploaded_files:
        # Read the image file from the uploaded file
        i_image = Image.open(uploaded_file)
        
        # Ensure the image is in RGB mode
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    # Extract pixel values for the model
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate output ids for the captions
    output_ids = model.generate(pixel_values, **gen_kwargs)

    # Decode the generated ids to get captions
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    
    return preds

# Streamlit UI
def main():
    st.title("Image Caption Generator")
    st.write("Upload images to generate captions!")

    # Load feature extractor and tokenizer
    feature_extractor, tokenizer = load_objects()

    # Load the model
    model = load_model()

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the appropriate device

    # Define generation kwargs (you can adjust these as needed)
    gen_kwargs = {
        "num_beams": 5,
        "max_length": 16,
        "early_stopping": True
    }

    # Upload multiple image files
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        # Display the uploaded images
        for uploaded_file in uploaded_files:
            st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)

        # Generate predictions for the uploaded images
        captions = predict_step(uploaded_files, model, feature_extractor, tokenizer, device, gen_kwargs)
        
        # Display the generated captions
        st.write("### Generated Captions:")
        for caption in captions:
            st.success(caption)

if __name__ == "__main__":
    main()
