import helper.config as cfg
import streamlit as st
import pandas as pd
import os, logging, torch

from pathlib import Path
from datasets import load_dataset
from haystack import Pipeline, Document
from haystack.utils import Secret, ComponentDevice
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.converters import HTMLToDocument, MarkdownToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.readers import ExtractiveReader
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.builders import AnswerBuilder, PromptBuilder
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.generators import OpenAIGenerator, HuggingFaceLocalGenerator
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.document_stores.in_memory import InMemoryDocumentStore
from tensorflow.python.ops.summary_ops_v2 import write
from tensorflow.tools.docs.doc_controls import header

st.set_page_config(page_title="Application #05", page_icon="🌹", layout="wide")
st.sidebar.title("🌹 RAG-Testpad")
st.sidebar.markdown(
    """This demo illustrates the capability of GenAI using Haystack and [**Streamlit**](https://docs.streamlit.io/develop/api-reference)
    frameworks. This testpad generates a series of AI tools based on the concept of RAG. Enjoy!"""
)
start_btn = st.sidebar.button(f"Click to **Start**", type="primary")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

if "HF_API_TOKEN" not in os.environ:
    os.environ["HF_API_TOKEN"] = cfg.dlreadtoken_key
    os.environ["HF_TOKEN"] = cfg.dlreadtoken_key

def rag_qna():
    # All filename in a specific folder
    filefolder = "datasets/"
    foldername = list(Path(filefolder).glob("**/*"))
    logging.info(f"[ai] foldername information: {foldername}")

    # Array list of filelist = ["file1.txt", "file2.txt", "file3.txt"]
    filelist1 = ["datasets/Malaysia_Corruption_Reports.txt", "datasets/Malaysia_Corruption_1MDB.txt"]
    filelist2 = ["datasets/Project_Gutenberg_LeoDaVin.txt"]
    filelist = filelist1
    logging.info(f"[ai] filelist information: {filelist}")

    # Get the data from a specific file only
    filename = "datasets/Malaysia_Corruption_Reports.txt"
    with open(filename, "r", encoding="utf-8") as filehandler:
        dataset = filehandler.read()
    filehandler.close()
    #dataset = load_dataset("text", data_files=filename, split="train")
    documents = [Document(content=dataset, meta={"name": filename})]
    logging.info(f"[ai] documents information: {documents}")

    # Default sample from public datasets
    #dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
    #documents = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

    st.write(f":green[dataset:] {documents[0].meta['name']}")

    template = """Given these documents, answer the question.
        Context:
        {% for doc in documents %}
        Document: {{ loop.index }} - File: {{ doc.meta['file_path'] }}
        {{ doc.content }}
        {% endfor %}
        Question: {{query}}
        Answer: """
    prompt_builder = PromptBuilder(template=template)

    model_embedder = "sentence-transformers/all-mpnet-base-v2" # better & larger model than below
    #model_embedder = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    #device_embedder = ComponentDevice.from_str("cuda:0") # train using accelerate GPU
    device_embedder = None

    document_store = InMemoryDocumentStore()
    indexing_pipeline = Pipeline()
    #indexing_pipeline.add_component(name="file_type_router", instance=FileTypeRouter(mime_types=["text/plain", "text/markdown", "application/pdf"]))
    indexing_pipeline.add_component(name="plain_converter", instance=TextFileToDocument(encoding="utf-8"))
    #indexing_pipeline.add_component(name="markdown_converter", instance=MarkdownToDocument())
    #indexing_pipeline.add_component(name="pypdf_converter", instance=PyPDFToDocument())
    indexing_pipeline.add_component(name="joiner", instance=DocumentJoiner())
    indexing_pipeline.add_component(name="cleaner", instance=DocumentCleaner())
    indexing_pipeline.add_component(name="splitter", instance=DocumentSplitter(split_by="word", split_length=200, split_overlap=50))
    indexing_pipeline.add_component(name="embedder", instance=SentenceTransformersDocumentEmbedder(model=model_embedder, device=device_embedder, progress_bar=True))
    indexing_pipeline.add_component(name="writer", instance=DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP))
    #indexing_pipeline.connect("file_type_router.text/plain", "plain_converter.sources")
    #indexing_pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
    #indexing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
    indexing_pipeline.connect("plain_converter", "joiner")
    #indexing_pipeline.connect("markdown_converter", "joiner")
    #indexing_pipeline.connect("pypdf_converter", "joiner")
    indexing_pipeline.connect("joiner", "cleaner")
    indexing_pipeline.connect("cleaner", "splitter")
    indexing_pipeline.connect("splitter", "embedder")
    indexing_pipeline.connect("embedder", "writer")

    #indexing_pipeline.run({"file_type_router": {"sources": foldername}})
    indexing_pipeline.run({"plain_converter": {"sources": filelist}})
    #indexing_pipeline.run(data={"joiner": {"documents": documents}})

    reader_answer = ExtractiveReader(no_answer=False)
    reader_answer.warm_up()
    #generator = OpenAIGenerator(model="gpt-4o", api_key=Secret.from_token(openai_key))
    generator1 = HuggingFaceLocalGenerator(
            model="HuggingFaceTB/SmolLM-1.7B-Instruct",
            task="text-generation",
            huggingface_pipeline_kwargs={"device_map": "auto",
                                         "model_kwargs": {"torch_dtype": torch.float16}},
            generation_kwargs={"max_new_tokens": 500, "temperature": 0.5, "do_sample": True})
    generator = HuggingFaceLocalGenerator(
            model="google/flan-t5-large",
            task="text2text-generation",
            huggingface_pipeline_kwargs={"device_map": "auto",
                                         "model_kwargs": {"torch_dtype": torch.float16}},
            generation_kwargs={"max_new_tokens": 500, "temperature": 0.5, "do_sample": True})
    # Start the Generator
    generator.warm_up()

    retriever_store = InMemoryEmbeddingRetriever(document_store=document_store, top_k=10)
    querying_pipeline = Pipeline()
    querying_pipeline.add_component(name="embedder", instance=SentenceTransformersTextEmbedder(model=model_embedder, device=device_embedder, progress_bar=True))
    querying_pipeline.add_component(name="retriever", instance=retriever_store)
    querying_pipeline.add_component(name="prompt_builder", instance=prompt_builder)
    querying_pipeline.add_component(name="reader", instance=reader_answer)
    querying_pipeline.add_component(name="generator", instance=generator)
    querying_pipeline.add_component(name="answer_builder", instance=AnswerBuilder())
    querying_pipeline.connect("embedder.embedding", "retriever.query_embedding")
    querying_pipeline.connect("retriever", "reader.documents") # Extractive QA Pipeline
    querying_pipeline.connect("retriever", "prompt_builder.documents")
    querying_pipeline.connect("prompt_builder", "generator") # Generative QA Pipeline
    querying_pipeline.connect("generator.replies", "answer_builder.replies")
    querying_pipeline.connect("retriever", "answer_builder.documents")

    # Extractive QA pipeline will consist of three components: an embedder, retriever, and reader.
    # Generative QA pipeline will consist of four components: an embedder, retriever, prompt_builder, and generator.
    # answer_builder component can be used by both Extractive and Generative QA pipeline to construct an output

    query1 = "List of corruption cases in Malaysia?"
    query2 = "Who was Pliny the Elder?"
    query = query1
    # Truncate the query (necessary if using api embedder since SPRM server has no GPU)
    words = query.split()
    truncated_words = words[:4000]
    query = ' '.join(truncated_words)

    data = {"embedder": {"text": query},
            "retriever": {"top_k": 5},
            "prompt_builder": {"query": query},
            "reader": {"query": query, "top_k": 3},
            "generator": {"generation_kwargs": {"max_new_tokens": 350}},
            "answer_builder": {"query": query}}
    #data["reader"] = {"query": query, "top_k": 3}
    answer = querying_pipeline.run(data=data,
                                   include_outputs_from = {"retriever", "reader", "generator", "answer_builder"} )

    with open("datasets/indexing_pipeline.yml", "w") as ifile:
        indexing_pipeline.dump(ifile)
    ifile.close()
    with open("datasets/querying_pipeline.yml", "w") as qfile:
        indexing_pipeline.dump(qfile)
    qfile.close()

    # Output formats according to type of pipelines
    #pipe["retriever"]["documents"][0].id/content/meta/score
    #pipe["reader"]["answers"][0].query/score/data/document/meta
    #pipe["generator"]["replies"][0]
    #pipe["answer_builder"]["answers"][0].data/query/documents/meta

    st.write(":red[**a1->**]", answer)
    st.write(":red[**a2->**]", answer["reader"]["answers"][0])
    st.write(":red[**a3->**] :blue[score=]", answer["reader"]["answers"][0].score, ":blue[, data=]", answer["reader"]["answers"][0].data)
    st.write(":red[**a4->**]", answer["generator"]["replies"][0])
    st.write(":red[**retriever->**]", answer["retriever"])
    st.write(":red[**reader->**]", answer["reader"])
    st.write(":red[**generator->**]", answer["generator"])
    st.write(":red[**answer_builder->**]", answer["answer_builder"])

if __name__ == '__main__':
    if start_btn:
        tab01, tab02, tab03, tab04, tab05 = st.tabs(
            ["👻 RAG-Q&A-01", "👻 RAG-Q&A-02", "👻 RAG-Q&A-03", "👻 RAG-Q&A-04", "👻 RAG-Q&A-local"])
        with tab01:
            st.subheader("RAG-Q&A-01")
        with tab02:
            st.subheader("RAG-Q&A-02")
        with tab03:
            st.subheader("RAG-Q&A-03")
        with tab04:
            st.subheader("RAG-Q&A-04")
        with tab05:
            st.subheader("RAG-Q&A_local_content")
            rag_qna()
