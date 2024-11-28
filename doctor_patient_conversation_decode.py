# -*- coding: utf-8 -*-
"""Doctor_Patient_Conversation_Decode"""

import streamlit as st
import whisper
import os
from io import BytesIO
from groq import Groq 

api_key = st.secrets["API_KEY"]

# Initialize Groq API client
client = Groq(api_key=api_key)

# Load the Whisper model
@st.cache_resource
def load_whisper_model():
    """
    Loads the Whisper model.
    Returns:
        The loaded Whisper model.
    """
    print("Loading Whisper model...")
    return whisper.load_model("base")

whisper_model = load_whisper_model()

# Function to transcribe audio
def transcribe_audio(audio_file):
    """
    Transcribes audio using Whisper and returns the text.
    Args:
        audio_file (BytesIO): Uploaded audio file.
    Returns:
        str: Transcribed text from the audio.
    """
    temp_audio_path = "temp_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_file.getbuffer())

    # Transcribe the audio file
    print(f"Transcribing audio: {temp_audio_path}")
    result = whisper_model.transcribe(temp_audio_path, fp16=False)
    transcription = result["text"]

    # Clean up the temporary file
    os.remove(temp_audio_path)
    print("Transcription completed.")
    return transcription

# Function to analyze transcription using GPT-4
def analyze_transcription(transcription):
    """
    Sends the transcription to GPT-4 for medical analysis.
    Args:
        transcription (str): Text from the transcription.
    Returns:
        str: GPT-4's analysis.
    """
    prompt = f"""
    The following is a conversation between a doctor and a patient:
    {transcription}

    Based on this conversation, provide:
    1. A possible prognosis for the patient.
    2. A detailed diagnosis of the condition.
    3. Medication recommendations or treatments for the patient.
    """
    print("Sending transcription to GPT-4 for analysis...")

    try:
        # Updated code to use the new OpenAI client interface
         #client = Groq(
         #api_key= st.secrets["API_KEY"]
         #)
        response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a medical assistant AI with expertise in prognosis, diagnosis, and medication recommendations."},
            {"role": "user", "content": prompt}
        ]
        )
        analysis = response['choices'][0]['message']['content']
        print("GPT-4 analysis received.")
        return analysis
    except Exception as e:
        print("Error during GPT-4 analysis:", str(e))
        return f"An error occurred during GPT-4 analysis: {str(e)}"

# Streamlit App Setup
st.title("Doctor-Patient Conversation Analysis")
st.write("Upload an audio file to transcribe and analyze a doctor-patient conversation.")

# File uploader for audio files
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg", "m4a"])

if uploaded_file is not None:
    # Step 1: Transcribe the audio
    with st.spinner("Transcribing the audio..."):
        transcription = transcribe_audio(uploaded_file)

    # Display the transcription
    st.subheader("Transcription:")
    st.write(transcription)

    # Step 2: Analyze the transcription
    with st.spinner("Analyzing the transcription..."):
        analysis = analyze_transcription(transcription)

    # Display the medical analysis
    st.subheader("Medical Analysis:")
    st.write(analysis)

    # Step 3: Provide download options for results
    st.download_button(
        label="Download Transcription",
        data=transcription,
        file_name="transcription.txt",
        mime="text/plain"
    )
    st.download_button(
        label="Download Medical Analysis",
        data=analysis,
        file_name="medical_analysis.txt",
        mime="text/plain"
    )
