import streamlit as st
from groq import Groq
import whisper
import os
# from io import BytesIO

# Fetch the API key from Streamlit Secrets
import os
os.environ['GROQ_API_KEY'] = st.secrets["GROQ_API_KEY"]
# print(f"API Key: {api_key}")
# Initialize Groq API client
client = Groq()

# @st.cache_resource
# def load_whisper_model():
#     """
#     Loads the Whisper model.
#     Returns:
#         The loaded Whisper model.
#     """
#     print("Loading Whisper model...")
#     return whisper.load_model("base")

# whisper_model = load_whisper_model()

# def transcribe_audio(audio_file):
#     """
#     Transcribes audio using Whisper and returns the text.
#     Args:
#         audio_file (BytesIO): Uploaded audio file.
#     Returns:
#         str: Transcribed text from the audio.
#     """
#     temp_audio_path = "temp_audio.wav"
#     with open(temp_audio_path, "wb") as f:
#         f.write(audio_file.getbuffer())

#     # Transcribe the audio file
#     print(f"Transcribing audio: {temp_audio_path}")
#     result = whisper_model.transcribe(temp_audio_path, fp16=False)
#     transcription = result["text"]

#     # Clean up the temporary file
#     os.remove(temp_audio_path)
#     print("Transcription completed.")
#     return transcription



# def analyze_transcription(transcription):
#     """
#     Sends the transcription to Groq for medical analysis.
#     Args:
#         transcription (str): Text from the transcription.
#     Returns:
#         str: Groq's analysis.
#     """
#     prompt = f"""
#     The following is a conversation between a doctor and a patient:
#     {transcription}

#     Based on this conversation, provide:
#     1. A possible prognosis for the patient.
#     2. A detailed diagnosis of the condition.
#     3. Medication recommendations or treatments for the patient.
#     """
#     print("Sending transcription to Groq for analysis...")

#     response = client.chat.completions.create(model="llama3-8b-8192",
#     messages=[
#     {"role": "system", "content": "You are a medical assistant AI with expertise in prognosis, diagnosis, and medication recommendations."},
#     {"role": "user", "content": prompt}
#     ]
#     )
#     analysis = response.choices[0].message.content
#     print("Groq analysis received.")
#     return analysis

# # Streamlit App Setup
# st.title("Doctor-Patient Conversation Analysis")
# st.write("Upload an audio file to transcribe and analyze a doctor-patient conversation.")

# # File uploader for audio files
# uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg", "m4a"])
# if uploaded_file is not None:
#     # Step 1: Transcribe the audio
#     with st.spinner("Transcribing the audio..."):
#         transcription = transcribe_audio(uploaded_file)

#     # Display the transcription
#     st.subheader("Transcription:")
#     st.write(transcription)

#     # Step 2: Analyze the transcription
#     with st.spinner("Analyzing the transcription..."):
#         analysis = analyze_transcription(transcription)

#     # Display the medical analysis
#     st.subheader("Medical Analysis:")
#     st.write(analysis)


### ------ version2
# Initialize chat history and uploaded files
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_audio_files" not in st.session_state:
    st.session_state.uploaded_audio_files = []

# Load the Whisper model
@st.cache_resource
def load_whisper_model():
    """
    Loads the Whisper model.
    Returns:
        The loaded Whisper model.
    """
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
    result = whisper_model.transcribe(temp_audio_path, fp16=False)
    transcription = result["text"]

    # Clean up the temporary file
    os.remove(temp_audio_path)
    return transcription

# Function to analyze transcription with Groq
def analyze_transcription(transcription):
    """
    Analyzes the transcription and generates a medical response using Groq.
    Args:
        transcription (str): Transcribed text.
    Returns:
        str: AI-generated medical analysis.
    """
    prompt = f"""
    The following is a conversation between a doctor and a patient:
    {transcription}

    Based on this conversation, provide:
    1. A possible prognosis for the patient.
    2. A detailed diagnosis of the condition.
    3. Medication recommendations or treatments for the patient.
    """
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a medical assistant AI with expertise in prognosis, diagnosis, and medication recommendations."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Streamlit page title
st.title("🩺 Medical Assistant")


# Sidebar for audio file uploads
st.sidebar.title("Upload Audio Files")
uploaded_audio_files = st.sidebar.file_uploader(
    "Drag and drop audio files here or click to browse",
    type=["mp3", "wav", "ogg", "m4a"],
    accept_multiple_files=True
)

# Process uploaded audio files
if uploaded_audio_files:
    for file in uploaded_audio_files:
        if file.name not in [f.name for f in st.session_state.uploaded_audio_files]:
            st.session_state.uploaded_audio_files.append(file)

# Display uploaded audio files
if st.session_state.uploaded_audio_files:
    st.sidebar.markdown("### Uploaded Audio Files:")
    for file in st.session_state.uploaded_audio_files:
        st.sidebar.markdown(f"- **{file.name}** ({file.size / 1024:.2f} KB)")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# Process uploaded audio files and integrate into chat
if st.session_state.uploaded_audio_files:
    for file in st.session_state.uploaded_audio_files:
        # Check if file is already processed
        if not any(f"Uploaded audio file: {file.name}" in msg["content"] for msg in st.session_state.chat_history):
            # Transcribe and analyze audio
            with st.spinner(f"Processing `{file.name}`..."):
                transcription = transcribe_audio(file)
                analysis = analyze_transcription(transcription)

            # Add file upload and analysis response to chat history
            st.session_state.chat_history.append({"role": "user", "content": f"Uploaded audio file: {file.name}"})
            st.session_state.chat_history.append({"role": "assistant", "content": f"**Transcription:**\n{transcription}"})
            st.session_state.chat_history.append({"role": "assistant", "content": f"**Analysis:**\n{analysis}"})

            # Display transcription and analysis in chat
            with st.chat_message("assistant"):
                st.markdown(f"**Transcription:**\n{transcription}")
            with st.chat_message("assistant"):
                st.markdown(f"**Analysis:**\n{analysis}")


# Handle user text input
if user_prompt := st.chat_input("Ask your questions?"):
    # Add user input to chat history
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Generate assistant response based on chat history
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "Answer questions based on previous analyses and chat history , give direct response and only related to the analysis, dont answer any other questions which is not related to the analysis"},
            *st.session_state.chat_history
        ]
    )
    assistant_response = response.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # Display assistant's response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)


