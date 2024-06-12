import streamlit as st
from spc_2_txt import Model
import whisper
from tempfile import NamedTemporaryFile

@st.cache_resource
def load_model():
    model = Model()
    return model

model = load_model()

audio = st.file_uploader("Upload an audio file", type=["mp3"])

col1, col2 = st.columns([1,3])
with col1:
    transcribe_button = st.button("Transcribe")
    
with col2:
    if audio is not None:
        if transcribe_button:
            with NamedTemporaryFile(suffix=".mp3", delete = False) as temp:
                temp.write(audio.getvalue())
                temp.seek(0)
                
                with st.spinner("Transcribing...\nPlease Wait🤗"):
                    t = model.tran_scribe(file_path=temp.name)
                    st.write("Transcribed Text: ", t)
