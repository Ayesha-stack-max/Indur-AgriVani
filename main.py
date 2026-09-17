import streamlit as st
import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
import os
from PIL import Image
from audio_recorder_streamlit import audio_recorder

api_key = os.environ.get("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("Please add your GEMINI_API_KEY to Render Environment Variables.")

st.set_page_config(page_title="Indur Agrivani", page_icon="🌾", layout="centered")
st.title("🌾 ఇందుర్ అగ్రివాణి (Indur Agrivani)")
st.subheader("Google AI-Powered Kisan Sahayak")

language_option = st.selectbox(
    "Choose your language / अपनी भाषा चुनें / మీ భాషను ఎంచుకోండి",
    ["Telugu", "Hindi", "English", "Hinglish"],
    index=0
)

LANG_MAP = {
    "Telugu": {
        "stt_lang": "te-IN", "tts_lang": "te",
        "prompt_instruction": "Answer this farmer's question in clear, polite, and pure Telugu language."
    },
    "Hindi": {
        "stt_lang": "hi-IN", "tts_lang": "hi",
        "prompt_instruction": "Answer this farmer's question in clear, polite, and standard Hindi language using Devanagari script."
    },
    "English": {
        "stt_lang": "en-IN", "tts_lang": "en",
        "prompt_instruction": "Answer this farmer's question in clear, polite, professional English."
    },
    "Hinglish": {
        "stt_lang": "hi-IN", "tts_lang": "hi",
        "prompt_instruction": "Answer this farmer's question in fluid, colloquial Hinglish (Hindi written using the English/Latin script). Do not use Devanagari script. Use standard English spelling for loan words."
    }
}
selected_config = LANG_MAP[language_option]

st.markdown("### 📸 Step 1: Check Damaged Leaves (Optional)")
leaf_image = st.file_uploader("Upload a photo of sick/damaged leaves or crops", type=["jpg", "jpeg", "png"])
if leaf_image:
    st.image(leaf_image, caption="Uploaded Leaf Image", use_container_width=True)

st.markdown("### 🎙️ Step 2: Ask Your Question")
audio_source = st.radio("Choose how to ask:", ["Record live speech 🎙️", "Upload recorded audio file 📁"])

audio_data_to_process = None
session_suffix = "uploaded"

if audio_source == "Record live speech 🎙️":
    st.write("Click the microphone icon below to speak:")
    audio_bytes = audio_recorder()
    if audio_bytes:
        audio_data_to_process = audio_bytes
        session_suffix = "live.wav"
else:
    audio_file = st.file_uploader(f"Upload {language_option} audio file (.wav format works best)", type=["wav", "mp3", "ogg", "m4a"])
    if audio_file is not None:
        audio_data_to_process = audio_file.getbuffer()
        session_suffix = audio_file.name

if audio_data_to_process is not None:
    st.info("🔄 Google AI is processing your input...")
    
    temp_filename = f"temp_{session_suffix}"
    with open(temp_filename, "wb") as f:
        f.write(audio_data_to_process)
        
    r = sr.Recognizer()
    try:
        with sr.AudioFile(temp_filename) as source:
            audio_source_data = r.record(source)
            
        with st.spinner(f"🎙️ Transcribing {language_option} speech..."):
            farmer_text = r.recognize_google(audio_source_data, language=selected_config["stt_lang"])
            
        st.success(f"🗣️ **Farmer Asked:** {farmer_text}")
        
        prompt = f"""
        You are 'Indur Agrivani', an expert agricultural AI assistant serving farmers in Nizamabad, Telangana. 
        {selected_config['prompt_instruction']}
        
        Provide professional, practical, low-cost answers to any farmer query regarding:
        1. WEATHER: Rain forecasting impacts, temperature damage, seasonal timing.
        2. CROPS & SOIL: Yield optimizations, fertilizer application rates (NPK, Urea), soil healthcare.
        3. LEAF & PLANT DISEASES: If a leaf image is provided alongside this prompt, analyze it visually. Identify pests, fungal spots, or mineral deficiencies and offer immediate solutions.
        4. GENERAL ISSUES: Irrigation problems, local crop methods for Rice, Turmeric, Sugarcane, Maize.

        Formatting Rules:
        - Keep the advice short, structured, and easy to follow.
        - Format with clean bullet points so the text-to-speech reads it aloud naturally.
        
        Farmer's Question: {farmer_text}
        """
        
        model = genai.GenerativeModel('gemini-3.6-flash') 
        
        with st.spinner("🧠 Gemini AI is analyzing your crops and query..."):
            if leaf_image:
                img = Image.open(leaf_image)
                response = model.generate_content([prompt, img])
            else:
                response = model.generate_content(prompt)
                
            bot_reply_text = response.text
            
        st.markdown(f"### 🤖 **Indur Agrivani Response:**\n{bot_reply_text}")
        
        with st.spinner("🔊 Generating audio response..."):
            tts = gTTS(text=bot_reply_text, lang=selected_config["tts_lang"])
            reply_filename = f"reply_{session_suffix}.mp3"
            tts.save(reply_filename)
        
        with open(reply_filename, "rb") as audio_out:
            st.audio(audio_out.read(), format="audio/mp3")
            st.balloons()

    except sr.UnknownValueError:
        st.error(f"❌ Google Speech Recognition could not understand the audio. Please speak clearly in {language_option}.")
    except sr.RequestError as e:
        st.error(f"❌ API Request failed; {e}")
    except Exception as e:
        st.error(f"❌ System error: {e}")
        
    finally:
        if os.path.exists(temp_filename): 
            os.remove(temp_filename)
        if 'reply_filename' in locals() and os.path.exists(reply_filename):
            os.remove(reply_filename)
