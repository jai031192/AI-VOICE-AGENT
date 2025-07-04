# 📁 voice_agent_ui.py
import streamlit as st
from chatbot3 import compiled_graph, ChatbotState, flattened_kb
from sarvam_stt_wrapped import RealTimeASR
from tts_module import speak_response_safe, tts_lock
import threading
import time

# --- Streamlit Page Setup ---
st.set_page_config(page_title="🚚 WheelsEye Voice Agent", layout="centered")
st.title("🚚 WheelsEye Voice Agent")

# --- Session State Setup ---
if "chat_state" not in st.session_state:
    st.session_state.chat_state: ChatbotState = {
        "user_input": "",
        "history": [],
        "response": "",
        "knowledge_base": flattened_kb,
        "is_first_message": True,
        "retrieved_context": [],
        "started": False
    }

# --- Voice Loop Threaded Logic ---
def conversation_loop(state: dict):
    while True:
        if tts_lock.locked():
            time.sleep(0.5)
            continue

        asr = RealTimeASR(silence_threshold=0.01, silence_duration=3.0, max_listen_time=10.0)
        print("🟢 Listening for user's input...")
        user_text = asr.listen_and_transcribe()

        if user_text.startswith("[ERROR]") or user_text.startswith("[FALLBACK]") or user_text.startswith("[INFO]"):
            speak_response_safe(user_text)
            continue

        state["user_input"] = user_text.strip()
        state["history"].append({"role": "operator", "text": user_text.strip()})
        state_new = compiled_graph.invoke(state)
        state.update(state_new)
        speak_response_safe(state["response"])

# --- Start Voice Agent ---
if not st.session_state.chat_state["started"]:
    if st.button("▶️ Start Voice Agent"):
        st.session_state.chat_state["started"] = True

        # Initial greeting
        greeting = "नमस्ते सर, मैं रवि बोल रहा हूँ WheelsEye से — आपकी गाड़ी लोड वाली है क्या?"
        st.session_state.chat_state["response"] = greeting
        st.session_state.chat_state["history"].append({"role": "rep", "text": greeting})

        speak_response_safe(greeting)

        # Start background conversation thread
        threading.Thread(target=conversation_loop, args=(st.session_state.chat_state,), daemon=True).start()
        st.stop()
    st.stop()

# --- Display Chat History ---
st.markdown("""
    <style>
    .bubble {
        padding: 10px 15px;
        margin: 8px 0;
        border-radius: 20px;
        max-width: 70%;
    }
    .user { background-color: #e0f7fa; margin-left: auto; text-align: right; }
    .rep { background-color: #e8f5e9; margin-right: auto; text-align: left; }
    </style>
""", unsafe_allow_html=True)

for msg in st.session_state.chat_state["history"]:
    cls = "user" if msg["role"] == "operator" else "rep"
    st.markdown(f'<div class="bubble {cls}"><b>{msg["role"]}</b>: {msg["text"]}</div>', unsafe_allow_html=True)

# --- Reset Button ---
if st.button("🔁 Reset"):
    st.session_state.clear()
    st.rerun()
