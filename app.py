import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from gtts import gTTS
from io import BytesIO

# --- Model Loading (Cached) ---
@st.cache_resource
def load_my_model():
    # This is your Fake vs. Real model
    model = load_model('Fake-currency.keras')
    return model

# @st.cache_resource
# def load_validation_model():
    # --- PLACEHOLDER ---
    # In a real app, you would load your *second* model here.
    # This model's job is to find 'currency' in any image.
    # model = load_model('currency_object_detector.h5')
    # return model

model = load_my_model()
# validation_model = load_validation_model()


# --- Preprocessing Function (No changes needed) ---
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# --- Prediction Function (No changes needed) ---
def predict_currency(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# --- NEW: Validation Function (Placeholder) ---
def is_image_a_currency_note(image):
    """
    Checks if the image contains a currency note.
    
    !!! THIS IS A PLACEHOLDER !!!
    To implement this, you need a separate object detection model
    (like YOLO or MobileNet) trained to identify currency notes.
    """
    
    # --- Example of real logic ---
    # preprocessed_for_validation = ... # Preprocess for your object detector
    # object_prediction = validation_model.predict(preprocessed_for_validation)
    # if 'currency_note' in object_prediction:
    #     return True
    # else:
    #     return False
    
    # For now, we return True so the rest of the code can run.
    # Replace this with your real validation logic.
    st.write("Validation check (placeholder)...")
    return True

# --- Text-to-Speech Function (No changes needed) ---
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.error(f"Error in TTS: {e}")
        return None

# --- NEW: Combined Processing Function ---
def process_and_predict(image):
    """
    Runs the full validation, prediction, and TTS pipeline.
    """
    # 1. Validate if the image is currency
    if not is_image_a_currency_note(image):
        # If not currency, show error and play error sound
        error_message = "Error. Please take a picture of currency only."
        st.error(error_message)
        audio_file = text_to_speech(error_message)
        if audio_file:
            st.audio(audio_file, format='audio/mp3', autoplay=True)
        return # Stop here

    # 2. If it is currency, proceed with fake detection
    st.write('Processing...')
    prediction = predict_currency(image)
    label = 'Fake Currency' if prediction[0][0] > 0.5 else 'Real Currency'
    
    if label == 'Fake Currency':
        st.error(f"Prediction: {label}")
    else:
        st.success(f"Prediction: {label}")
    
    # 3. Play the result automatically
    st.write("Playing audio result...")
    audio_file = text_to_speech(label)
    if audio_file:
        # Added autoplay=True
        st.audio(audio_file, format='audio/mp3', autoplay=True)

# --- Streamlit UI ---
st.title("Fake Currency Detection")

tab1, tab2 = st.tabs(["📁 Upload from Gallery", "📸 Use Camera"])

# --- Tab 1: Gallery Upload ---
with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, channels="BGR", caption="Uploaded Image")
        
        # --- AUTOMATIC CALL (No button) ---
        process_and_predict(image)

# --- Tab 2: Camera Input ---
with tab2:
    camera_file = st.camera_input("Take a picture")
    
    if camera_file is not None:
        file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, channels="BGR", caption="Captured Image")
        
        # --- AUTOMATIC CALL (No button) ---
        process_and_predict(image)