import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import io

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="music_genre_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define genre labels
genres = {1: "metal", 3: "blues", 5: "classical", 8: "country", 0: "disco", 
          7: "hiphop", 9: "jazz", 6: "rock", 4: "reggae", 2: "pop"}

def process_audio(audio_file, sr=22050, duration=30):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=sr, duration=duration)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Pad or truncate to match the input shape of the model
    if mfccs.shape[1] > 259:
        mfccs = mfccs[:, :259]
    else:
        mfccs = np.pad(mfccs, ((0, 0), (0, 259 - mfccs.shape[1])), mode='constant')
    
    return mfccs.T

st.title('Music Genre Classification')

# Display supported genres
st.write("Supported genres:")
supported_genres = ['blues', 'metal', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'pop', 'reggae', 'rock']
st.write(", ".join(supported_genres))

uploaded_file = st.file_uploader("Choose a music file", type=['wav', 'mp3', 'ogg'])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    # Convert uploaded file to audio array
    audio_bytes = uploaded_file.read()
    
    try:
        # Process audio
        mfccs = process_audio(io.BytesIO(audio_bytes))
        
        # Prepare input data
        input_data = np.expand_dims(mfccs, axis=0).astype(np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get predicted genre
        predicted_genre = genres[np.argmax(output_data)]
        st.success(f"Predicted Genre: {predicted_genre}")
        
        # Display confidence for each genre
        st.write("Confidence for each genre:")
        for index, confidence in enumerate(output_data[0]):
            genre = genres[index]
            st.write(f"{genre}: {confidence:.2f}")
    
    except Exception as e:
        st.error(f"An error occurred while processing the audio: {str(e)}")
        st.error("Please make sure you've uploaded a valid audio file.")
else:
    st.write("Please upload an audio file to classify its genre.")
