import helper.config as cfg
import streamlit as st
import pandas as pd
import logging, random, time, os

import whisper
import sounddevice as sd
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from scipy.io.wavfile import write

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
        logging.info(f"Transcription: {result['text']}")
        return result["text"]
    else:
        raise ValueError("Provide either an audio file path or set use_microphone=True")


# Function for Text-to-Speech using gTTS
def text_to_speech(text, output_file="temp/output.mp3"):
    logging.info("Converting text to speech...")
    tts = gTTS(text=text, lang='en')
    tts.save(output_file)
    logging.info(f"Speech saved to {output_file}")

    # Play the generated speech
    audio = AudioSegment.from_mp3(output_file)
    play(audio)

# Main function to handle both STT and TTS
def speechtextconverter():
    # Path to your audio file (for STT)
    audio_file = "temp/output.wav"  # Replace with your audio file path

    # Step 1: Convert Speech to Text using Whisper
    transcribed_text = speech_to_text(audio_file=audio_file)
    st.write(f"STT Transcription: {transcribed_text}")
    # Step 2: Convert Text to Speech using gTTS
    text_to_speech(transcribed_text)


def main():
    tab01, tab02, tab03, tab04, tab05 = st.tabs(["👻 Other", "👻 Other", "👻 Other", "👻 Other", "👻 Other"])
    with tab01:
        st.subheader("Other-01")
        result = speech_to_text(use_microphone=True)
        st.write(f"STT Transcription: {result}")
    with tab02:
        st.subheader("Other-02")
        speechtextconverter()
        st.write(f"Speech Text Analyser")
    with tab03:
        st.subheader("Other-03")
    with tab04:
        st.subheader("Other-04")
    with tab05:
        st.subheader("Other-05")
        htmltodoc()

if __name__ == '__main__':
    start_btn = st.sidebar.button(f"Click to **Start**", type="primary", use_container_width=True)
    if start_btn is True:
        st.title("Speech Text Analyser")
        with st.container(border=True):
            main()
