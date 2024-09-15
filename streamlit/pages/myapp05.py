import helper.config as cfg
import streamlit as st
import pandas as pd
import torch, logging, random, time, os

from openai import OpenAI
from haystack.core.pipeline import Pipeline
from haystack.dataclasses import Document
from haystack.utils import ComponentDevice
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
st.sidebar.title("🪻 AI Chatbots")
st.sidebar.markdown(
    """This demo illustrates a combination of different AI chatbots. Try with different
    combination of AI python frameworks with Streamlit platform. Enjoy!"""
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = cfg.Application_key
if "HF_API_TOKEN" not in os.environ:
    os.environ["HF_API_TOKEN"] = cfg.dlreadtoken_key
    os.environ["HF_TOKEN"] = cfg.dlreadtoken_key

def chatgpt():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            try:
                stream = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[{"role": m["role"], "content": m["content"]}
                              for m in st.session_state.messages],
                    stream=True,)
                response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                logging.info(f"Error: :red[**{e}**]")
                st.write(f"Error: :red[**{e}**]")

def simplechat():
    fetcher = LinkContentFetcher()
    converter = HTMLToDocument()
    prompt_template = """
    According to the contents of this website:
    {% for document in documents %}
      {{document.content}}
    {% endfor %}
    Answer the given question: {{query}}
    Answer:
    """
    prompt_builder = PromptBuilder(template=prompt_template)
    llm = OpenAIGenerator()

    pipeline = Pipeline()
    pipeline.add_component("fetcher", fetcher)
    pipeline.add_component("converter", converter)
    pipeline.add_component("prompt", prompt_builder)
    pipeline.add_component("llm", llm)

    pipeline.connect("fetcher.streams", "converter.sources")
    pipeline.connect("converter.documents", "prompt.documents")
    pipeline.connect("prompt.prompt", "llm.prompt")

    try:
        result = pipeline.run({"fetcher": {"urls": ["https://haystack.deepset.ai/overview/quick-start"]},
                               "prompt": {"query": "Which components do I need for a RAG pipeline?"}})
        st.markdown(result["llm"]["replies"][0])
    except Exception as e:
        logging.info(f"Error: :red[**{e}**]")
        st.write(f"Error: :red[**{e}**]")

def get_generative_answer(query_pipeline, query):
  results = query_pipeline.run({
      "text_embedder": {"text": query},
      "prompt_builder": {"query": query}
    }
  )
  answer = results["generator"]["replies"][0]
  return answer

def ragchat():
    uploaded_file = st.file_uploader(":blue[**Choose a excel file**]", type=['xls', 'xlsx'], accept_multiple_files=False)
    if uploaded_file is not None:
        content_file = pd.read_fwf(uploaded_file)
        content_data = [Document(content=content_file)]
        st.write(content_data)
        # In memory document store
        document_store = InMemoryDocumentStore()
        #
        #🚅 Components
        #- splitter: DocumentSplitter
        #- embedder: SentenceTransformersDocumentEmbedder
        #- writer: DocumentWriter
        #🛤️ Connections
        #- splitter.documents -> embedder.documents(List[Document])
        #- embedder.documents -> writer.documents(List[Document])
        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="word", split_length=200))
        indexing_pipeline.add_component("embedder",
            SentenceTransformersDocumentEmbedder(
                model="Snowflake/snowflake-arctic-embed-l", # good embedding model: https://huggingface.co/Snowflake/snowflake-arctic-embed-l
                device=None,                                # load the model on GPU = ComponentDevice.from_str("cuda:0")
            ))
        indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
        # connect the components
        indexing_pipeline.connect("splitter", "embedder")
        indexing_pipeline.connect("embedder", "writer")
        #
        indexing_pipeline.run({"splitter": {"documents": content_data}})
        #
        # RAF prompt template
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
        #
        #
        generator = HuggingFaceLocalGenerator(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            huggingface_pipeline_kwargs={"device_map": "auto",
                                         "model_kwargs": {"load_in_4bit": True,
                                                          "bnb_4bit_use_double_quant": True,
                                                          "bnb_4bit_quant_type": "nf4",
                                                          "bnb_4bit_compute_dtype": torch.bfloat16}},
            generation_kwargs={"max_new_tokens": 500})
        #
        generator.warm_up()
        #
        #
        query_pipeline = Pipeline()
        query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(
                model="Snowflake/snowflake-arctic-embed-l", # good embedding model: https://huggingface.co/Snowflake/snowflake-arctic-embed-l
                device=ComponentDevice.from_str("cuda:0"),  # load the model on GPU
                prefix="Represent this sentence for searching relevant passages: ", # as explained in the model card (https://huggingface.co/Snowflake/snowflake-arctic-embed-l#using-huggingface-transformers), queries should be prefixed
            ))
        query_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=5))
        query_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
        query_pipeline.add_component("generator", generator)
        # connect the components
        query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        query_pipeline.connect("retriever.documents", "prompt_builder.documents")
        query_pipeline.connect("prompt_builder", "generator")
        #
        if prompt := st.chat_input("What is up?"):
            # q = "Who won the Best Picture Award in 2024?"
            answer = get_generative_answer(query_pipeline, prompt)
            st.write(answer)
    else:
        st.markdown(":red[**Pls upload a excel file...**]")

def main():
    tab01, tab02, tab03, tab04, tab05 = st.tabs(["👻 Other", "👻 Other", "👻 RAG AI", "👻 OpenAI", "👻 SimpleChat"])
    with tab01:
        st.subheader("Chatbot-01")
    with tab02:
        st.subheader("Chatbot-02")
    with tab03:
        ragchat()
    with tab04:
        chatgpt()
    with tab05:
        simplechat()

if __name__ == '__main__':
    st.title("Simple AI Chatbots")
    main()
