import streamlit as st
from spc_2_txt import Model
import whisper
# import ffmpeg
# from summary import Summarize
# from translate import Translate
from tempfile import NamedTemporaryFile
# from langdetect import detect
from test import hi

@st.cache_resource
def load_model():
    model = Model()
    return model

# @st.cache_resource
# def load_real_model():
#     model = whisper.load_model('medium')
#     return model

# test = load_real_model()
model = load_model()

# test = load_model()

audio = st.file_uploader("Upload an audio file", type=["mp3"])

col1, col2 = st.columns([1,3])
with col1:
    transcribe_button = st.button("Transcribe")
    translate_button = st.button("Translate")
    # summarize_button = st.button("Summarize")
with col2:
    if audio is not None:
        if transcribe_button:
            with NamedTemporaryFile(suffix=".mp3", delete = False) as temp:
                temp.write(audio.getvalue())
                temp.seek(0)
                # transcribe = model_.tran_scribe(file_path = temp.name)
                # st.write(test.hello(temp.name))
                # test = load_model()
                st.write(temp.name)
                # t = test.hello(file_path = temp.name)
                with st.spinner("Transcribing...\nPlease Wait🤗"):
                    t = model.tran_scribe(file_path=temp.name)
                    st.write("Transcribed Text: ", t)
                # st.write(type(temp.name))
                
                # src_lang = detect(transcribe) #length check it...
                # st.write("Transcribed Text: ", transcribe)
                
        if translate_button:
            tgt_langs = ['af', 'ar', 'bg', 'bn', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'fa', 'fi', 'fr', 'gu', 'he', 'hi', 'hr', 'hu', 'id', 'it', 'ja', 'kn', 'ko', 'lt', 'lv', 'mk', 'ml', 'mr', 'ne', 'nl', 'no', 'pa', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'so', 'sq', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi']
            tgt_lang = st.selectbox("Select Target Language: ", tgt_langs)
            st.write("You selected", tgt_lang)
            # translated_text = Translate.translate_(src = src_lang, tgt = tgt_lang, text = transcribe)
        #     st.write("Translated Text: ", translated_text)
            
        # if summarize_button:
        #     summary = Summarize.summarize(text = transcribe)
        #     st.write("Summarized Text", summary)