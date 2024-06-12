import whisper

class Model:
    def __init__(self) -> None:
        self.model = whisper.load_model("medium")
        
    def tran_scribe(self, file_path: str):
        tcription = self.model.transcribe(file_path, fp16 = False)
        text = tcription['text']
        return text
        
    def detect_lang(self, file_path):
        audio = whisper.load_audio(file_path)
        audio = whisper.pad_or_trim(audio)
        
        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        
        # detect the spoken language
        _, probs = self.model.detect_language(mel)
        
        detected_audio = max(probs, key=probs.get)
        return detected_audio