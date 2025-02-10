import helper.config as cfg
import streamlit as st
import pandas as pd
import logging, random, time, os

import tempfile
import torch
import whisper
import sounddevice as sd
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from scipy.io.wavfile import write
from transformers import pipeline

#from haystack import Pipeline
#from haystack import Document
#from haystack.utils import Secret, ComponentDevice
from haystack.components.fetchers import LinkContentFetcher
#from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.converters import HTMLToDocument, TextFileToDocument
#from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
#from haystack.components.builders import PromptBuilder
#from haystack.components.preprocessors import DocumentSplitter
#from haystack.components.generators import OpenAIGenerator, HuggingFaceLocalGenerator
#from haystack.components.writers import DocumentWriter
#from haystack.document_stores.in_memory import InMemoryDocumentStore

st.set_page_config(page_title="Application #05", page_icon="🪻", layout="wide")
st.sidebar.title("🪻 Speech Text Analyser")
st.sidebar.markdown(
    """This demo illustrates a combination of different Speech Text Analyser. Try with
    different combination of AI python frameworks with Streamlit platform. Enjoy!"""
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger("haystack").setLevel(logging.INFO)

def upload_file():
    with st.form("upload-documents", clear_on_submit=True, border=True):
        uploaded_file = st.file_uploader(":blue[**Choose a audio file**]", type=['mp3', 'wav'], accept_multiple_files=False)
        submitted = st.form_submit_button("Confirm Upload")
        if (submitted is True) and (uploaded_file is not None):
            st.audio(uploaded_file, format="audio/wav")

            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_filepath = temp_file.name

            # Clean up temporary file
            os.remove(temp_filepath)
            logging.info(f"Uploaded_file: {temp_filepath}")
            return temp_filepath

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

def htmltodoc():
    urls = [
        "https://techcrunch.com/2023/04/27/pinecone-drops-100m-investment-on-750m-valuation-as-vector-database-demand-grows/",
        "https://techcrunch.com/2023/04/27/replit-funding-100m-generative-ai/",
        "https://www.cnbc.com/2024/06/12/mistral-ai-raises-645-million-at-a-6-billion-valuation.html",
        "https://techcrunch.com/2024/01/23/qdrant-open-source-vector-database/",
        "https://www.intelcapital.com/anyscale-secures-100m-series-c-at-1b-valuation-to-radically-simplify-scaling-and-productionizing-ai-applications/",
        "https://techcrunch.com/2023/04/28/openai-funding-valuation-chatgpt/",
        "https://techcrunch.com/2024/03/27/amazon-doubles-down-on-anthropic-completing-its-planned-4b-investment/",
        "https://techcrunch.com/2024/01/22/voice-cloning-startup-elevenlabs-lands-80m-achieves-unicorn-status/",
        "https://techcrunch.com/2023/08/24/hugging-face-raises-235m-from-investors-including-salesforce-and-nvidia",
        "https://www.prnewswire.com/news-releases/ai21-completes-208-million-oversubscribed-series-c-round-301994393.html",
        "https://techcrunch.com/2023/03/15/adept-a-startup-training-ai-to-use-existing-software-and-apis-raises-350m/",
        "https://www.cnbc.com/2023/03/23/characterai-valued-at-1-billion-after-150-million-round-from-a16z.html",
    ]
    fetcher = LinkContentFetcher()
    streams = fetcher.run(urls=urls)["streams"]
    converter = HTMLToDocument()
    docs = converter.run(sources=streams)
    st.write(docs)

def main():
    tab01, tab02, tab03, tab04, tab05 = st.tabs(["👻 STTmic", "👻 STTS", "👻 STT1", "👻 STT2", "👻 Html2Doc"])
    with tab01:
        st.subheader("STT from Microphone")
        tab01_btn = st.button(label="Click to **Start**", key="tab01_btn")
        if tab01_btn is True:
            result = speech_to_text(use_microphone=True)
            st.write(f"STT Transcription: {result}")
    with tab02:
        st.subheader("Speech Text Converter")
        tab02_btn = st.button(label="Click to **Start**", key="tab02_btn")
        if tab02_btn is True:
            speechtextconverter()
    with tab03:
        st.subheader("STT-01")
        file = upload_file()
        st.write(f"Uploaded file: {file}")
        #speechtotext01()
    with tab04:
        st.subheader("STT-02")
        speechtotext02()
    with tab05:
        st.subheader("Html2Doc")
        htmltodoc()

if __name__ == '__main__':
    st.title("Speech Text Analyser")
    main()