# 📁 sarvam_stt_wrapped.py
import sounddevice as sd
import numpy as np
import wave
import tempfile
import os
from sarvamai import SarvamAI

class RealTimeASR:
    def __init__(self, duration=5, fs=16000):
        self.duration = duration
        self.fs = fs
        self.client = SarvamAI(api_subscription_key="133e4b4d-02be-4fb8-af46-406dec1f5425")

    def record_to_temp_wav(self):
        print("🎤 Recording...")
        audio = sd.rec(int(self.duration * self.fs), samplerate=self.fs, channels=1, dtype='int16')
        sd.wait()
        print("✅ Done recording.")

        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(temp_wav.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.fs)
            wf.writeframes(audio.tobytes())
        return temp_wav.name

    def listen_and_transcribe(self):
        wav_path = self.record_to_temp_wav()
        print("🧠 Transcribing with Sarvam AI...")
        with open(wav_path, "rb") as f:
            response = self.client.speech_to_text.transcribe(
                file=f,
                model="saarika:v2.5",
                language_code="hi-IN"
            )
        os.remove(wav_path)
        return response.transcript

# Optional test
if __name__ == "__main__":
    asr = RealTimeASR()
    text = asr.listen_and_transcribe()
    print("📝 You said:", text)