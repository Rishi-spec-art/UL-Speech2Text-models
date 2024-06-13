import streamlit as st
import sounddevice as sd
import numpy as np
import whisper
import threading

# Load the Whisper model
@st.cache_resource
def load_model():
    model = whisper.load_model("base")
    return model

# Variables to control recording
is_recording = False
audio_data = []
samplerate = 16000

# Callback function to capture audio data
def audio_callback(indata, frames, time, status):
    global audio_data
    if status:
        st.error(f"Error: {status}", icon="🚨")
    audio_data.extend(indata[:, 0].tolist())

# Function to start recording
def start_recording():
    global is_recording, audio_data
    is_recording = True
    audio_data = []
    threading.Thread(target=record_audio).start()

# Function to stop recording
def stop_recording():
    global is_recording
    is_recording = False

# Function to record audio
def record_audio():
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate):
        while is_recording:
            sd.sleep(100)

# Function to transcribe audio
def transcribe_audio():
    global audio_data
    if len(audio_data) == 0:
        return "No audio data to transcribe."
    
    audio_array = np.array(audio_data, dtype=np.float32)
    st.write(f"Audio data captured: {len(audio_array)} samples")

    # Transcribe using Whisper
    model = load_model()
    result = model.transcribe(audio_array, fp16=False)  # Use fp16=False for CPU inference
    return result["text"]

# Streamlit UI
st.title("Real-Time Speech-to-Text with Whisper")

if st.button("Start Recording"):
    start_recording()
    st.write("Recording...")

if st.button("Stop Recording"):
    stop_recording()
    st.write("Recording stopped.")

if st.button("Transcribe"):
    transcription = transcribe_audio()
    st.write("Transcription:")
    st.write(transcription)

