import helper.config as cfg
import streamlit as st
import pandas as pd
import logging, random, time, os

from haystack import Pipeline
from haystack import Document
from haystack.utils import Secret, ComponentDevice
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.converters import HTMLToDocument, TextFileToDocument
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.builders import PromptBuilder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.generators import OpenAIGenerator, HuggingFaceLocalGenerator
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

st.set_page_config(page_title="Application #05", page_icon="🪻", layout="wide")
st.sidebar.title("🪻 Simple Code")
st.sidebar.markdown(
    """This demo illustrates a combination of different AI chatbots. Try with different
    combination of AI python frameworks with Streamlit platform. Enjoy!"""
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

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
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import HTMLToDocument

fetcher = LinkContentFetcher()
streams = fetcher.run(urls=urls)["streams"]
converter = HTMLToDocument()
docs = converter.run(sources=streams)
st.write(docs)

def main():
    tab01, tab02, tab03, tab04, tab05 = st.tabs(["👻 Other", "👻 Other", "👻 Other", "👻 Other", "👻 Other"])
    with tab01:
        st.subheader("Other-01")
    with tab02:
        st.subheader("Other-02")
    with tab03:
        st.subheader("Other-03")
    with tab04:
        st.subheader("Other-04")
    with tab05:
        st.subheader("Other-05")

if __name__ == '__main__':
    st.title("Simple Code")
    main()
