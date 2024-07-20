import streamlit as st
import numpy as np
import librosa
from keras.models import load_model
import io
from st_audiorec import st_audiorec
import tensorflow as tf

# Load and compile the model
@st.cache_resource
def load_and_compile_model():
    model = load_model('music_genre_model.h5')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_and_compile_model()

# Define genre labels
genres = {1: "metal", 3: "blues", 5: "classical", 8: "country", 0: "disco", 
          7: "hiphop", 9: "jazz", 6: "rock", 4: "reggae", 2: "pop"}

@tf.function(input_signature=[tf.TensorSpec(shape=[None, 259, 13], dtype=tf.float32)])
def predict_genre(mfccs):
    return model(mfccs)

def process_audio(audio_file, sr=22050, duration=30):
    y, sr = librosa.load(audio_file, sr=sr, duration=duration)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    if mfccs.shape[1] > 259:
        mfccs = mfccs[:, :259]
    else:
        mfccs = np.pad(mfccs, ((0, 0), (0, 259 - mfccs.shape[1])), mode='constant')
    return mfccs.T

st.title('Music Genre Classification')

# Add radio buttons for choosing input method
input_method = st.radio("Choose input method:", ('Upload File', 'Record Audio'), horizontal=True)

def process_and_predict(audio_bytes):
    try:
        mfccs = process_audio(io.BytesIO(audio_bytes))
        mfccs_tensor = tf.convert_to_tensor(np.expand_dims(mfccs, axis=0), dtype=tf.float32)
        prediction = predict_genre(mfccs_tensor)
        predicted_genre = genres[np.argmax(prediction)]
        st.success(f"Predicted Genre: {predicted_genre}")
    except Exception as e:
        st.error(f"An error occurred while processing the audio: {str(e)}")
        st.error("Please make sure you've provided a valid audio file or recording.")

if input_method == 'Upload File':
    uploaded_file = st.file_uploader("Choose a music file", type=['wav', 'mp3', 'ogg'])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        audio_bytes = uploaded_file.read()
        process_and_predict(audio_bytes)

elif input_method == 'Record Audio':
    st.write("Click the microphone to start recording. Click again to stop.")
    audio_bytes = st_audiorec()
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        process_and_predict(audio_bytes)

st.write("Supported genres:")
supported_genres = ['blues', 'metal', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'pop', 'reggae', 'rock']
st.write(", ".join(supported_genres))