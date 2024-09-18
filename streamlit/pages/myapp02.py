import helper.config as cfg
import streamlit as st
import pandas as pd
import tempfile as tf
import torch, time, logging, shutil, os

from haystack.core.pipeline import Pipeline
from haystack.utils import ComponentDevice
from haystack.dataclasses import Document
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.converters import HTMLToDocument, TextFileToDocument, MarkdownToDocument, PyPDFToDocument
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.builders import PromptBuilder
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.generators import OpenAIGenerator, HuggingFaceLocalGenerator
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.document_stores.in_memory import InMemoryDocumentStore

st.set_page_config(page_title="Application #02", page_icon="🌾", layout="wide")
st.sidebar.title("🌾 Query Assistant")
st.sidebar.markdown(
    """This GenAI illustrates a combination of local contents with public AI models to produce
    a Question & Answer AI Assistant based on user uploaded (:rainbow[**multiple**]) documents. Enjoy!"""
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

if "HF_API_TOKEN" not in os.environ:
    os.environ["HF_API_TOKEN"] = cfg.dlreadtoken_key
    os.environ["HF_TOKEN"] = cfg.dlreadtoken_key

# Embedder model used
embedder_model0 = "Snowflake/snowflake-arctic-embed-l" # good embedding model: https://huggingface.co/Snowflake/snowflake-arctic-embed-l
embedder_model1 = "sentence-transformers/all-mpnet-base-v2"
device_model0 = ComponentDevice.from_str("cuda:0")  # load the model on GPU
device_model1 = None
embedder_model = embedder_model1
device_model = device_model1

def upload_files(cleanup = False):
    if "temp_documents" not in st.session_state:
        st.session_state.temp_documents = [] # array list of files to process
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = tf.mkdtemp() # use a temporary store folder
    with st.form("upload-documents", clear_on_submit=True, border=True):
        uploaded_files = st.file_uploader(":blue[**Choose multiple text/pdf files**]", type=['txt', 'pdf'], accept_multiple_files=True)
        submitted = st.form_submit_button("Confirm Upload")
        if (submitted is True) and (len(uploaded_files) != 0):
            st.session_state.temp_documents = [] # clear it before load new list
            for upload_file in uploaded_files:
                temp_file = os.path.join(st.session_state.temp_dir, upload_file.name)
                logging.info(f"[ai] file object: {upload_file}")
                logging.info(f"[ai] file path: {temp_file}")
                st.session_state.temp_documents.append(temp_file)
                with open(mode="w+b", file=temp_file) as fn:
                    fn.write(upload_file.getvalue())
                    fn.close()
            uploaded_files.clear()
        if len(st.session_state.temp_documents) != 0:
            st.write(st.session_state.temp_documents)
        if cleanup:
            shutil.rmtree(st.session_state.temp_dir)
    logging.info(f"[ai] uploaded_files: {uploaded_files}")
    logging.info(f"[ai] temp_documents: {st.session_state.temp_documents}")
    return st.session_state.temp_documents

def upload_files1(cleanup = False):
    if "temp_documents" not in st.session_state:
        st.session_state.temp_documents = [] # array list of files to process
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = tf.mkdtemp() # use a temporary store folder
    uploaded_files = st.file_uploader(":blue[**Choose multiple text/pdf files**]", type=['txt', 'pdf'], accept_multiple_files=True)
    if len(uploaded_files) != 0:
        st.session_state.temp_documents = [] # clear it before load new list
        for upload_file in uploaded_files:
            temp_file = os.path.join(st.session_state.temp_dir, upload_file.name)
            logging.info(f"[ai] file object: {upload_file}")
            logging.info(f"[ai] file path: {temp_file}")
            st.session_state.temp_documents.append(temp_file)
            with open(mode="w+b", file=temp_file) as fn:
                fn.write(upload_file.getvalue())
                fn.close()
        uploaded_files.clear()
    else:
        st.markdown(":red[**Pls upload few text/pdf files...**]")
    if len(st.session_state.temp_documents) != 0:
        st.write(st.session_state.temp_documents)
    if cleanup:
        shutil.rmtree(st.session_state.temp_dir)
    logging.info(f"[ai] uploaded_files: {uploaded_files}")
    logging.info(f"[ai] temp_documents: {st.session_state.temp_documents}")
    return st.session_state.temp_documents

def prompt_syntax():
    # Define a Template Prompt
    prompt_template = """Using the information contained in the context, give a comprehensive answer to the question.
    If the answer cannot be deduced from the context, do not give an answer.
    Context:
      {% for doc in documents %}
      {{ doc.content }} URL:{{ doc.meta['file_path'] }}
      {% endfor %};
      Question: {{query}}
      Answer: """
    return prompt_template

def index_xpipeline(document_store):
    # Building the Index Pipeline
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(name="file_type_router", instance=FileTypeRouter(mime_types=["text/plain", "text/markdown", "application/pdf"]))
    indexing_pipeline.add_component(name="plain_converter", instance=TextFileToDocument(encoding="utf-8"))
    indexing_pipeline.add_component(name="markdown_converter", instance=MarkdownToDocument())
    indexing_pipeline.add_component(name="pypdf_converter", instance=PyPDFToDocument(converter=None))
    indexing_pipeline.add_component(name="joiner", instance=DocumentJoiner())
    indexing_pipeline.add_component(name="cleaner", instance=DocumentCleaner())
    indexing_pipeline.add_component(name="splitter", instance=DocumentSplitter(split_by="word", split_length=200, split_overlap=50))
    indexing_pipeline.add_component(name="embedder", instance=SentenceTransformersDocumentEmbedder(model=embedder_model, device=device_model, progress_bar=True))
    indexing_pipeline.add_component(name="writer", instance=DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP))
    # connect the components
    indexing_pipeline.connect("file_type_router.text/plain", "plain_converter.sources")
    indexing_pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
    indexing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
    indexing_pipeline.connect("plain_converter", "joiner")
    indexing_pipeline.connect("markdown_converter", "joiner")
    indexing_pipeline.connect("pypdf_converter", "joiner")
    indexing_pipeline.connect("joiner.documents", "cleaner.documents")
    indexing_pipeline.connect("cleaner.documents", "splitter.documents")
    indexing_pipeline.connect("splitter.documents", "embedder.documents")
    indexing_pipeline.connect("embedder.documents", "writer.documents")
    return indexing_pipeline

def query_xpipeline(document_store, prompt_template, generator):
    # Build the Query Pipeline
    querying_pipeline = Pipeline()
    querying_pipeline.add_component("embedder", SentenceTransformersTextEmbedder(model=embedder_model, device=device_model, progress_bar=True))
    querying_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=5))
    querying_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
    querying_pipeline.add_component("generator", generator)
    # connect the components
    querying_pipeline.connect("embedder.embedding", "retriever.query_embedding")
    querying_pipeline.connect("retriever.documents", "prompt_builder.documents")
    querying_pipeline.connect("prompt_builder", "generator")
    return querying_pipeline

def get_generative_answer(query_pipeline, query):
    # Let's ask some questions!
    results = query_pipeline.run({
        "embedder": {"text": query},
        "retriever": {"top_k": 5},
        "prompt_builder": {"query": query},
        "generator": {"generation_kwargs": {"max_new_tokens": 350}}
        })
    logging.info(f"[ai] generator results: {results}")
    answer = results["generator"]["replies"][0]
    return answer

def rag_chatbot():
    # Fetch the Text Data
    sources_data = upload_files()
    if len(sources_data) != 0:
        # In memory document store
        document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
        # Building the Index Pipeline
        indexing_pipeline = index_xpipeline(document_store)
        indexing_pipeline.run(data={"file_type_router": {"sources": sources_data}})
        #
        # Define a Template Prompt
        prompt_template = prompt_syntax()
        # Initialize a Generator
        #generator = HuggingFaceLocalGenerator(model="HuggingFaceTB/SmolLM-1.7B-Instruct",
        #                                      task="text-generation",
        #                                      huggingface_pipeline_kwargs={"device_map": "auto",
        #                                                                   "model_kwargs": {"torch_dtype": torch.float16}},
        #                                      generation_kwargs={"max_new_tokens": 500, "temperature": 0.5, "do_sample": True})
        # The LLM we choose is Zephyr 7B Beta, a fine-tuned version of Mistral 7B V.01 that focuses on helpfulness and outperforms many larger models on the MT-Bench and AlpacaEval benchmarks; the model was fine-tuned by the Hugging Face team
        #generator = HuggingFaceLocalGenerator(model="HuggingFaceH4/zephyr-7b-beta",
        #                                      task="text-generation",
        #                                      huggingface_pipeline_kwargs={"device_map": "auto",
        #                                                                   "model_kwargs": {"load_in_4bit": True,
        #                                                                                    "bnb_4bit_use_double_quant": True,
        #                                                                                    "bnb_4bit_quant_type": "nf4",
        #                                                                                    "bnb_4bit_compute_dtype": torch.bfloat16}},
        #                                      generation_kwargs={"max_new_tokens": 350})
        generator = HuggingFaceLocalGenerator(model="google/flan-t5-large",
                                              task="text2text-generation",
                                              huggingface_pipeline_kwargs={"device_map": "auto",
                                                                           "model_kwargs": {"torch_dtype": torch.float16}},
                                              generation_kwargs={"max_new_tokens": 500, "temperature": 0.5, "do_sample": True})
        # Start the Generator
        generator.warm_up()
        # Build the Query Pipeline
        querying_pipeline = query_xpipeline(document_store, prompt_template, generator)
        #
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        #
        if prompt := st.chat_input("What is up?"):
            with st.chat_message("user"):
                # Truncate the query (necessary if using api embedder since system has no GPU)
                words = prompt.split()
                truncated_words = words[:4000]
                prompt = ' '.join(truncated_words)
                st.session_state.messages.append({"role": "user", "content": prompt})
                logging.info(f"[ai] user query: {prompt}")
                st.write(prompt)
            with st.chat_message("assistant"):
                try:
                    response = get_generative_answer(querying_pipeline, prompt)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    logging.info(f"[ai] ai response: {response}")
                    st.write(response)
                except Exception as e:
                    logging.error(f"Error: :red[**{e}**]")

def main():
    rag_chatbot()

if __name__ == '__main__':
    #st.title("Query Assistant")
    reset_btn = st.sidebar.button(f"Click to **Reset**", type="primary", use_container_width=True)
    if reset_btn is True:
        st.session_state.temp_documents = []
        st.session_state.clear()
        st.cache_data.clear()
        st.cache_resource.clear()
    st.sidebar.image(image="helper/eco-friendly.png", caption=None, use_column_width="always")
    main()
