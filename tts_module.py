import os
import base64
import threading
from io import BytesIO
import tempfile
from sarvamai import SarvamAI
import simpleaudio as sa  # ✅ Replaces ffplay-based playback

# --- Sarvam API Setup ---
SARVAM_API_KEY = "133e4b4d-02be-4fb8-af46-406dec1f5425"  # 🔁 Replace with your actual key
sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

# --- Global TTS Lock ---
tts_lock = threading.Lock()

# --- Optional custom temp folder ---
CUSTOM_TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_audio")
os.makedirs(CUSTOM_TEMP_DIR, exist_ok=True)

# --- Generate and Play Speech with Sarvam AI ---
def speak(text: str, async_play: bool = True):
    def _generate_and_play():
        try:
            print(f"🗣️ Speaking (Sarvam): {text}")
            print(f"📂 Temp Dir Being Used: {CUSTOM_TEMP_DIR}")

            response = sarvam_client.text_to_speech.convert(
                text=text,
                target_language_code="hi-IN",
                model="bulbul:v2",
                speaker="hitesh",  # ✅ or "meera", "karun", "anushka"
                pitch=0.2,
                pace=1.2,
                loudness=1,
                speech_sample_rate=22050,
                enable_preprocessing=True
            )

            b64_audio = response.audios[0]
            audio_bytes = base64.b64decode(b64_audio)

            # ✅ Save audio to temp .wav file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=CUSTOM_TEMP_DIR) as tmpfile:
                tmpfile.write(audio_bytes)
                tmpfile_path = tmpfile.name

            # ✅ Play audio using simpleaudio
            wave_obj = sa.WaveObject.from_wave_file(tmpfile_path)
            play_obj = wave_obj.play()
            play_obj.wait_done()

            # ✅ Clean up the temp file
            os.remove(tmpfile_path)

        except Exception as e:
            print(f"❌ TTS Error (Sarvam): {e}")

    if async_play:
        threading.Thread(target=lambda: _safe_tts(_generate_and_play)).start()
    else:
        _safe_tts(_generate_and_play)

# --- Locking wrapper to avoid TTS overlap ---
def _safe_tts(fn):
    if tts_lock.locked():
        print("🔒 Waiting for current audio...")
    with tts_lock:
        fn()

# --- Safe external call ---
def speak_response_safe(text: str):
    speak(text, async_play=True)

# --- Exports ---
__all__ = ["speak", "speak_response_safe", "tts_lock"]
