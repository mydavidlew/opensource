import helper.config as cfg
import streamlit as st
import pandas as pd
import torch, random, time, logging, os

from io import StringIO
from haystack import Pipeline
from haystack import Document
from haystack.utils import Secret, ComponentDevice
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.converters import HTMLToDocument, TextFileToDocument
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.builders import PromptBuilder
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.generators import OpenAIGenerator, HuggingFaceLocalGenerator
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.document_stores.in_memory import InMemoryDocumentStore

st.set_page_config(page_title="Application #01", page_icon="🌸", layout="wide")
st.sidebar.title("🌸 Query Assistant")
st.sidebar.markdown(
    """This GenAI illustrates a combination of local content with public AI models to produce
    a Question & Answer AI Assistant based on user uploaded (:rainbow[**single**]) document. Enjoy!"""
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger("haystack").setLevel(logging.INFO)

if "HF_API_TOKEN" not in os.environ:
    os.environ["HF_API_TOKEN"] = cfg.dlreadtoken_key
    os.environ["HF_TOKEN"] = cfg.dlreadtoken_key

# Embedder model used
embedder_model0 = "Snowflake/snowflake-arctic-embed-l" # good embedding model: https://huggingface.co/Snowflake/snowflake-arctic-embed-l
embedder_model1 = "sentence-transformers/all-mpnet-base-v2"
embedder_model2 = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
device_model0 = ComponentDevice.from_str("cuda:0")  # load the model on GPU
device_model1 = None
embedder_model = embedder_model1
device_model = device_model1

def upload_file():
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0
    if "content_data" not in st.session_state:
        st.session_state.content_data = None
    # Fetch the Text Data
    with st.form("upload-documents", clear_on_submit=True, border=True):
        uploaded_file = st.file_uploader(":blue[**Choose a text file**]", type=['txt'], accept_multiple_files=False)
        submitted = st.form_submit_button("Confirm Upload")
        if (submitted is True) and (uploaded_file is not None):
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()
            #st.write(f":red[**bytes_data**]", bytes_data)
            # To read file
            #reads_data = st.session_state.uploaded_file.read()
            #st.write(f":red[**reads_data**]", reads_data)
            # To convert to a string based IO:
            #stringio = StringIO(st.session_state.uploaded_file.getvalue().decode("utf-8"))
            #st.write(f":red[**stringio**]", stringio)
            # To read file as string:
            #string_data = stringio.read()
            #st.write(f":red[**string_data**]", string_data)
            # Can be used wherever a "file-like" object is accepted:
            #dataframe = pd.read_fwf(st.session_state.uploaded_file)
            #st.write(f":red[**dataframe**]", dataframe)
            st.session_state.content_data = [Document(content=bytes_data.decode('utf-8'), meta={"name": uploaded_file.name, "type": uploaded_file.type, "size": uploaded_file.size, "url": uploaded_file._file_urls})]
        if st.session_state.content_data is not None:
            st.write(f"[ File: :orange[**{st.session_state.content_data[0].meta['name']}**] --- Type: :orange[**{st.session_state.content_data[0].meta['type']}**] --- Size: :orange[**{st.session_state.content_data[0].meta['size']}**] bytes ]")
    logging.info(f"[ai] uploaded_file: {uploaded_file}")
    logging.info(f"[ai] content_data: {st.session_state.content_data}")
    return st.session_state.content_data

def upload_file1():
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    # Fetch the Text Data
    st.session_state.uploaded_file = st.file_uploader(":blue[**Choose a text file**]", type=['txt'], accept_multiple_files=False, key=st.session_state.file_uploader_key)
    if st.session_state.uploaded_file is not None:
        # To read file as bytes:
        bytes_data = st.session_state.uploaded_file.getvalue()
        #st.write(f":red[**bytes_data**]", bytes_data)
        # To read file
        #reads_data = st.session_state.uploaded_file.read()
        #st.write(f":red[**reads_data**]", reads_data)
        # To convert to a string based IO:
        #stringio = StringIO(st.session_state.uploaded_file.getvalue().decode("utf-8"))
        #st.write(f":red[**stringio**]", stringio)
        # To read file as string:
        #string_data = stringio.read()
        #st.write(f":red[**string_data**]", string_data)
        # Can be used wherever a "file-like" object is accepted:
        #dataframe = pd.read_fwf(st.session_state.uploaded_file)
        #st.write(f":red[**dataframe**]", dataframe)
        content_data = [Document(content=bytes_data.decode('utf-8'), meta={"name": st.session_state.uploaded_file.name, "type": st.session_state.uploaded_file.type, "size": st.session_state.uploaded_file.size, "url": st.session_state.uploaded_file._file_urls})]
        logging.info(f"[ai] uploaded_file: {st.session_state.uploaded_file}")
        logging.info(f"[ai] content_data: {content_data}")
        return content_data
    else:
        st.markdown(":red[**Pls upload a text file...**]")
        return None

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
    #prompt_builder = PromptBuilder(template=prompt_template)
    return prompt_template

def index_xpipeline(document_store):
    # Building the Index Pipeline
    indexing_pipeline = Pipeline()
    #indexing_pipeline.add_component(name="joiner", instance=DocumentJoiner())
    indexing_pipeline.add_component(name="cleaner", instance=DocumentCleaner())
    indexing_pipeline.add_component(name="splitter", instance=DocumentSplitter(split_by="word", split_length=200, split_overlap=50))
    indexing_pipeline.add_component(name="embedder", instance=SentenceTransformersDocumentEmbedder(model=embedder_model, device=device_model, progress_bar=True))
    indexing_pipeline.add_component(name="writer", instance=DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP))
    # connect the components
    #indexing_pipeline.connect("joiner.documents", "cleaner.documents")
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
    data = {"embedder": {"text": query},
            "retriever": {"top_k": 5},
            "prompt_builder": {"query": query},
            "generator": {"generation_kwargs": {"max_new_tokens": 350}}}
    results = query_pipeline.run(data=data)
    logging.info(f"[ai] generator results: {results}")
    answer = results["generator"]["replies"][0]
    return answer

def test_chatbot():
    # Sample documents to process
    content_data = [Document(content="My name is DavidLew and I live in Berlin"),
                    Document(content="I saw a black horse running in my house"),
                    Document(content="Germany has many big cities including Berlin")]
    #
    # In memory document store
    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    # Building the Index Pipeline
    indexing_pipeline = index_xpipeline(document_store)
    #indexing_pipeline.run(data={"joiner": {"documents": content_data}})
    indexing_pipeline.run(data={"cleaner": {"documents": content_data}})
    #
    # Define a Template Prompt
    prompt_template = prompt_syntax()
    # Initialize a Generator
    # generator = HuggingFaceLocalGenerator(
    #    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    #    huggingface_pipeline_kwargs={"device_map": "auto",
    #                                 "model_kwargs": {"load_in_4bit": True,
    #                                                  "bnb_4bit_use_double_quant": True,
    #                                                  "bnb_4bit_quant_type": "nf4",
    #                                                  "bnb_4bit_compute_dtype": torch.bfloat16}},
    #    generation_kwargs={"max_new_tokens": 500})
    generator = HuggingFaceLocalGenerator(
        model="HuggingFaceTB/SmolLM-1.7B-Instruct",
        huggingface_pipeline_kwargs={"device_map": "auto",
                                     "model_kwargs": {}},
        generation_kwargs={"max_new_tokens": 500, "do_sample": True})
    # Start the Generator
    generator.warm_up()
    # Build the Query Pipeline
    querying_pipeline = query_xpipeline(document_store, prompt_template, generator)
    #
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

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
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                response = get_generative_answer(querying_pipeline, prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                logging.info(f"[ai] ai response: {response}")
                st.markdown(response)
            except Exception as e:
                logging.error(f"Error: :red[**{e}**]")

def rag_chatbot():
    # Fetch the Text Data
    content_data = upload_file()
    if content_data is not None:
        #st.write(content_data)
        #
        # In memory document store
        document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
        # Building the Index Pipeline
        indexing_pipeline = index_xpipeline(document_store)
        #indexing_pipeline.run(data={"joiner": {"documents": content_data}})
        indexing_pipeline.run(data={"cleaner": {"documents": content_data}})
        #
        # Define a Template Prompt
        prompt_template = prompt_syntax()
        # Initialize a Generator
        #generator = HuggingFaceLocalGenerator(model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        #                                      huggingface_pipeline_kwargs={"device_map": "auto",
        #                                                                   "model_kwargs": {"load_in_4bit": True,
        #                                                                                   "bnb_4bit_use_double_quant": True,
        #                                                                                   "bnb_4bit_quant_type": "nf4",
        #                                                                                   "bnb_4bit_compute_dtype": torch.bfloat16}},
        #                                      generation_kwargs={"max_new_tokens": 500})
        #generator = HuggingFaceLocalGenerator(model="HuggingFaceTB/SmolLM-1.7B-Instruct",
        #                                      task="text-generation",
        #                                      huggingface_pipeline_kwargs={"device_map": "auto",
        #                                                                   "model_kwargs": {"torch_dtype": torch.float16}},
        #                                      generation_kwargs={"max_new_tokens": 500, "temperature": 0.5, "do_sample": True})
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
            st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

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
                logging.info(f"[ai] user query: {prompt}")
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.markdown(prompt) # st.chat_message("user").markdown(prompt)
            # get_generative_answer("Who won the Best Picture Award in 2024?")
            # get_generative_answer("What was the box office performance of the Best Picture nominees?")
            # get_generative_answer("What was the reception of the ceremony")
            # get_generative_answer("Can you name some of the films that got multiple nominations?")
            # --- unrelated question: let's see how our RAG pipeline performs.
            # get_generative_answer("Audioslave was formed by members of two iconic bands. Can you name the bands and discuss the sound of Audioslave in comparison?")
            with st.chat_message("assistant"):
                try:
                    response = get_generative_answer(querying_pipeline, prompt)
                    logging.info(f"[ai] ai response: {response}")
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.markdown(response) # st.chat_message("assistant").markdown(response)
                except Exception as e:
                    logging.error(f"Error: :red[**{e}**]")

def main():
    #test_chatbot()
    rag_chatbot()

if __name__ == '__main__':
    #st.title("Query Assistant")
    reset_btn = st.sidebar.button(f"Click to **Reset**", type="primary", use_container_width=True)
    if reset_btn is True:
        st.session_state.file_uploader_key += 1
        st.session_state.content_data = None
        st.session_state.uploaded_file = None
        st.session_state.clear()
        st.cache_data.clear()
        st.cache_resource.clear()
    st.sidebar.image(image="helper/eco-friendly.png", caption=None, use_column_width="always")
    main()
