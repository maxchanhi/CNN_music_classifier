import streamlit as st
import numpy as np
import librosa
from keras.models import load_model
import io

# Load and compile the model
model = load_model('music_genre_model.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define genre labels (adjust these to match your model's output)
genres ={1:"metal",3:"blues",5:"classical",8:"country",0:"disco",7:"hiphop",9:"jazz",6:"rock",4:"reggae",2:"pop"}

def process_audio(audio_file, sr=22050, duration=30):
    y, sr = librosa.load(audio_file, sr=sr, duration=duration)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    if mfccs.shape[1] > 259:
        mfccs = mfccs[:, :259]
    else:
        mfccs = np.pad(mfccs, ((0, 0), (0, 259 - mfccs.shape[1])), mode='constant')
    
    return mfccs.T

st.title('Music Genre Classification')

uploaded_file = st.file_uploader("Choose a music file", type=['wav', 'mp3', 'ogg'])


if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    # Convert uploaded file to audio array
    audio_bytes = uploaded_file.read()
    
    try:
        # Process audio
        mfccs = process_audio(io.BytesIO(audio_bytes))
        # Make prediction
        prediction = model.predict(np.expand_dims(mfccs, axis=0))
        predicted_genre = genres[np.argmax(prediction)]
        st.success(f"Predicted Genre: {predicted_genre}")
        
    
    except Exception as e:
        st.error(f"An error occurred while processing the audio: {str(e)}")
        st.error("Please make sure you've uploaded a valid audio file.")
else:
    st.write("Supported genres:")
    supported_genres = ['blues', 'metal', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'pop', 'reggae', 'rock']
    st.write(", ".join(supported_genres))
