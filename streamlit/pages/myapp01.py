import streamlit as st
import pandas as pd
import torch, random, time, os

from io import StringIO
from haystack import Pipeline
from haystack.utils import ComponentDevice
from haystack.dataclasses import Document
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.converters import HTMLToDocument, TextFileToDocument
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.builders import PromptBuilder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.generators import OpenAIGenerator, HuggingFaceLocalGenerator
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from tensorflow.python.ops.summary_ops_v2 import write
from tensorflow.tools.docs.doc_controls import header

st.set_page_config(page_title="Application #01", page_icon="🌸", layout="wide")
st.sidebar.title("🌸 Application #01")
st.sidebar.markdown(
    """This demo illustrates a combination of geospatial data visualisation, plotting and animation with 
    [**Streamlit**](https://docs.streamlit.io/develop/api-reference). We're generating a bunch of random numbers 
    in a loop for around 5 seconds. Enjoy!"""
)

# HuggingFace
dlreadtoken_key = "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"
dlwritetoken_key = "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"

if "HF_API_TOKEN" not in os.environ:
    os.environ["HF_API_TOKEN"] = dlreadtoken_key

def upload_files():
    # Fetch the Text Data
    uploaded_file = st.file_uploader(":blue[**Choose a text file**]", type=['txt'], accept_multiple_files=False)
    if uploaded_file is not None:
        # To read file as bytes:
        #bytes_data = uploaded_file.getvalue()
        #st.write(f":red[**bytes_data**]", bytes_data)
        # To read file
        #reads_data = uploaded_file.read()
        #st.write(f":red[**reads_data**]", reads_data)
        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        #st.write(f":red[**stringio**]", stringio)
        # To read file as string:
        string_data = stringio.read()
        #st.write(f":red[**string_data**]", string_data)
        # Can be used wherever a "file-like" object is accepted:
        #dataframe = pd.read_fwf(uploaded_file)
        #st.write(f":red[**dataframe**]", dataframe)
        content_data = [Document(content=string_data, meta={"name": uploaded_file.name, "type": uploaded_file.type, "size": uploaded_file.size, "url": uploaded_file._file_urls})]
        uploaded_file.close()
        return content_data
    else:
        st.markdown(":red[**Pls upload a text file...**]")
        return None

def prompt_syntax():
    # Define a Template Prompt
    prompt_template = """
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>


    Using the information contained in the context, give a comprehensive answer to the question.
    If the answer cannot be deduced from the context, do not give an answer.

    Context:
      {% for doc in documents %}
      {{ doc.content }} URL:{{ doc.meta['url'] }}
      {% endfor %};
      Question: {{query}}<|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>


    """
    prompt_builder = PromptBuilder(template=prompt_template)
    return prompt_template, prompt_builder

def index_xpipeline(document_store):
    # Building the Index Pipeline
    # Embedder model used
    ##embedder_model0 = "Snowflake/snowflake-arctic-embed-l"  # good embedding model: https://huggingface.co/Snowflake/snowflake-arctic-embed-l
    ##device_model0 = ComponentDevice.from_str("cuda:0")  # load the model on GPU
    embedder_model1 = "Snowflake/snowflake-arctic-embed-l" # "sentence-transformers/all-MiniLM-L6-v2"
    device_model1 = None
    #
    # 🚅 Components
    # - splitter: DocumentSplitter
    # - embedder: SentenceTransformersDocumentEmbedder
    # - writer: DocumentWriter
    # 🛤️ Connections
    # - splitter.documents -> embedder.documents(List[Document])
    # - embedder.documents -> writer.documents(List[Document])
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="word", split_length=200, split_overlap=0))
    indexing_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model=embedder_model1, device=device_model1))
    indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
    # connect the components
    indexing_pipeline.connect("splitter", "embedder")
    indexing_pipeline.connect("embedder", "writer")
    return indexing_pipeline

def query_xpipeline(document_store, prompt_template, generator):
    # Build the Query Pipeline
    # Embedder model used
    ##embedder_model0 = "Snowflake/snowflake-arctic-embed-l" # good embedding model: https://huggingface.co/Snowflake/snowflake-arctic-embed-l
    ##device_model0 = ComponentDevice.from_str("cuda:0") # load the model on GPU
    embedder_model1 = "Snowflake/snowflake-arctic-embed-l" # "sentence-transformers/all-MiniLM-L6-v2"
    device_model1 = None
    #
    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model=embedder_model1, device=device_model1,
                                prefix="Represent this sentence for searching relevant passages: "))  # as explained in the model card (https://huggingface.co/Snowflake/snowflake-arctic-embed-l#using-huggingface-transformers), queries should be prefixed
    query_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=5))
    query_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
    query_pipeline.add_component("generator", generator)
    # connect the components
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    query_pipeline.connect("retriever.documents", "prompt_builder.documents")
    query_pipeline.connect("prompt_builder", "generator")
    return query_pipeline

def get_generative_answer(query_pipeline, query):
    # Let's ask some questions!
    results = query_pipeline.run({
        "text_embedder": {"text": query},
        "prompt_builder": {"query": query}
        })
    answer = results["generator"]["replies"][0]
    return answer

def test_chatbot():
    # In memory document store
    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    # Define a Template Prompt
    prompt_template, prompt_builder = prompt_syntax()
    #
    content_data = [Document(content="My name is Wolfgang and I live in Berlin"),
                    Document(content="I saw a black horse running"),
                    Document(content="Germany has many big cities")]
    #
    # Building the Index Pipeline
    indexing_pipeline = index_xpipeline(document_store)
    indexing_pipeline.run(data={"splitter": {"documents": content_data}})
    #
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
        generation_kwargs={"max_new_tokens": 500})
    # Start the Generator
    generator.warm_up()
    #
    # Build the Query Pipeline
    querying_pipeline = query_xpipeline(document_store, prompt_template, generator)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    #
    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                response = get_generative_answer(querying_pipeline, prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)
            except Exception as e:
                st.write(f"Error: :red[**{e}**]")

def rag_chatbot():
    # Fetch the Text Data
    content_data = upload_files()
    if content_data:
        #st.write(content_data)
        #
        # In memory document store
        document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
        # Define a Template Prompt
        prompt_template, prompt_builder = prompt_syntax()
        #
        # Building the Index Pipeline
        indexing_pipeline = index_xpipeline(document_store)
        indexing_pipeline.run(data={"splitter": {"documents": content_data}})
        #
        # Initialize a Generator
        #generator = HuggingFaceLocalGenerator(
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
            generation_kwargs={"max_new_tokens": 500})
        # Start the Generator
        generator.warm_up()
        #
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
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.markdown(prompt)
            # get_generative_answer("Who won the Best Picture Award in 2024?")
            # get_generative_answer("What was the box office performance of the Best Picture nominees?")
            # get_generative_answer("What was the reception of the ceremony")
            # get_generative_answer("Can you name some of the films that got multiple nominations?")
            # --- unrelated question: let's see how our RAG pipeline performs.
            # get_generative_answer("Audioslave was formed by members of two iconic bands. Can you name the bands and discuss the sound of Audioslave in comparison?")
            with st.chat_message("assistant"):
                try:
                    response = get_generative_answer(querying_pipeline, prompt)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.write(response)
                except Exception as e:
                    st.write(f"Error: :red[**{e}**]")

def main():
    test_chatbot()

if __name__ == '__main__':
    st.title("Hello world")
    st.cache_resource.clear()
    main()