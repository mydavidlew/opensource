import streamlit as st
import sounddevice as sd
import logging, tempfile, time, os
import torch, whisper
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from scipy.io.wavfile import write
from transformers import pipeline

st.set_page_config(page_title="Application #04", page_icon="🪻", layout="wide")
st.sidebar.title("🪻 Speech Analytic")
st.sidebar.markdown(
    """This demo illustrates a combination of different Speech Text Analyser. Try with
    different combination of AI python frameworks with Streamlit platform. Enjoy!"""
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger("haystack").setLevel(logging.INFO)

# Initialize components
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")

@st.cache_resource
def load_transformer_pipeline():
    return pipeline("text2text-generation", model="t5-small")

def upload_file():
    #with st.form("upload-documents", clear_on_submit=True, border=True):
        uploaded_file = st.file_uploader(":blue[**Choose a audio file**]", type=['mp3', 'wav'], accept_multiple_files=False)
        #submitted = st.form_submit_button("Confirm Upload")
        submitted = st.button("Confirm Upload", type="secondary")
        if (submitted is True) and (uploaded_file is not None):
            st.audio(uploaded_file, format="audio/wav")

            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_filepath = temp_file.name
            logging.info(f"Sound_file: {temp_filepath}")

            # Transcribe with Whisper
            st.subheader("Basic Transcription")
            transcription = speech_to_text(temp_filepath)
            st.write(transcription)

            # Process with Transformer
            st.subheader("Enhanced with Transformer")
            summarizer = load_transformer_pipeline()
            summary = summarizer("summarize: " + transcription, max_length=150)[0]['generated_text']
            st.write(summary)

            # Add download button
            st.download_button(
                label="Download Transcription",
                data=transcription,
                file_name="output.txt",
                mime="text/plain"
            )
            logging.info(f"Transcript_file: {temp_filepath}")

            # Clean up temporary file
            logging.info(f"Clearing cache: {temp_filepath}")
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)

# Function for Speech-to-Text using Whisper
def speech_to_text(audio_file=None, use_microphone=False):
    if use_microphone:
        logging.info("Recording audio...")
        # Record audio from microphone
        filename = "temp/output.wav"
        fs = 16000  # Sample rate
        duration = 5  # Duration in seconds
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        write(filename, fs, audio)  # Save as temporary file
        logging.info(f"Recording file: {filename}")
        audio_file = filename

    if audio_file:
        logging.info("Transcribing audio...")
        # Load Whisper model (choose 'tiny', 'base', 'small', 'medium', or 'large')
        model = whisper.load_model("small")
        # Transcribe audio file
        result = model.transcribe(audio_file)
        transcribed_text = result["text"]
        logging.info(f"Transcription: {transcribed_text}")
        return transcribed_text
    else:
        raise ValueError("Provide either an audio file path or set use_microphone=True")

# Function for Text-to-Speech using gTTS
def text_to_speech(text, lang='en', output_file="temp/output.mp3"):
    logging.info("Converting text to speech...")
    tts = gTTS(text=text, lang=lang, slow=False, lang_check=True)
    tts.save(output_file)
    logging.info(f"Speech saved to {output_file}")
    return output_file

def play_audio_file(audio_file):
    # Play the generated speech
    audio = None
    if audio_file.endswith(".mp3"):
        audio = AudioSegment.from_mp3(audio_file)
    elif audio_file.endswith(".wav"):
        audio = AudioSegment.from_wav(audio_file)
    if audio:
        logging.info(f"Play SoundFile: {audio_file}")
        play(audio)

# Optional: Convert MP3 to WAV if needed (Whisper works with WAV files)
def convert_mp3_to_wav(mp3_file, wav_file):
    logging.info(f"Converting {mp3_file} to {wav_file}")
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_file, format="wav")

# Main function to handle both STT and TTS
def speechtextconverter():
    # To create a Python program that performs Text-to-Speech (TTS) and Speech-to-Text (STT) using gTTS (Google
    # Text-to-Speech), Whisper (for STT), and leveraging the transformer architecture, we can follow these steps:
    # - Speech-to-Text (STT) : Use OpenAI's Whisper model to convert speech into text.
    # - Text-to-Speech (TTS) : Use gTTS to convert text into speech.
    #
    # Path to your audio file (for STT)
    audio_file = "temp/sample-0.mp3"  # Replace with your audio file path

    # Step 1: Convert Speech to Text using Whisper
    transcribed_text = speech_to_text(audio_file=audio_file)
    st.write(f"STT Transcription: {transcribed_text}")
    # Step 2: Convert Text to Speech using gTTS
    speech_file = text_to_speech(transcribed_text, lang='en')
    st.write(f"TTS SoundFile: {speech_file}")
    # Step 3: Play the sound file
    play_audio_file(speech_file)
    st.write(f"Play SoundFile: {speech_file}")

def speechtotext01():
    logging.info("Transcribing audio 01...")
    # Load Whisper model (choose 'tiny', 'base', 'small', 'medium', or 'large')
    model = whisper.load_model("small")
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio("temp/sample-0.mp3")
    audio = whisper.pad_or_trim(audio)
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    # detect the spoken language
    _, probs = model.detect_language(mel)
    logging.info(f"Detected language: {max(probs, key=probs.get)}")
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    # print the recognized text
    print("Transcription:", result.text)

def speechtotext02():
    logging.info("Transcribing audio 02...")
    # Load the pre-trained Whisper model
    stt_pipeline = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-small",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    # Process an audio file
    audio_path = "temp/sample-0.mp3"
    output = stt_pipeline(audio_path)["text"]
    print("Transcription:", output)

def main():
    tab01, tab02, tab03, tab04, tab05 = st.tabs(["👻 STTmic", "👻 STTS", "👻 STT1", "👻 STT2", "👻 Test"])
    with tab01:
        st.subheader("STT from Microphone")
        tab01_btn = st.button(label="Click to **Start**", key="tab01_btn")
        if tab01_btn is True:
            logging.info(f"Tab1: STT from Microphone")
            result = speech_to_text(use_microphone=True)
            st.write(f"STT Transcription: {result}")
    with tab02:
        st.subheader("Speech Text Converter")
        tab02_btn = st.button(label="Click to **Start**", key="tab02_btn")
        if tab02_btn is True:
            logging.info(f"Tab2: Speech Text Converter")
            speechtextconverter()
    with tab03:
        st.subheader("STT-01")
        tab03_btn = st.button(label="Click to **Start**", key="tab03_btn")
        if tab03_btn is True:
            logging.info(f"Tab3: STT-01")
            speechtotext01()
    with tab04:
        st.subheader("STT-02")
        tab04_btn = st.button(label="Click to **Start**", key="tab04_btn")
        if tab04_btn is True:
            logging.info(f"Tab4: STT-02")
            speechtotext02()
    with tab05:
        st.subheader("Test")
        logging.info(f"Tab5: Test")
        upload_file()

if __name__ == '__main__':
    st.title("Speech Text Analyser")
    main()