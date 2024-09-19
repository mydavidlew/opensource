import helper.config as cfg
import streamlit as st
import pandas as pd
import os, logging, torch
import tempfile as tf

from pathlib import Path
from datasets import load_dataset
from haystack.core.pipeline import Pipeline
from haystack.dataclasses import Document
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

st.set_page_config(page_title="Application #06", page_icon="🌹", layout="wide")
st.sidebar.title("🌹 RAG-Testpad")
st.sidebar.markdown(
    """This demo illustrates the capability of GenAI using Haystack and [**Streamlit**](https://docs.streamlit.io/develop/api-reference)
    frameworks. This testpad generates a series of AI tools based on the concept of RAG. Enjoy!"""
)
#start_btn = st.sidebar.button(f"Click to **Start**", type="primary")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

if "HF_API_TOKEN" not in os.environ:
    os.environ["HF_API_TOKEN"] = cfg.dlreadtoken_key
    os.environ["HF_TOKEN"] = cfg.dlreadtoken_key

def rag_qna_single():
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
    #generator1 = HuggingFaceLocalGenerator(
    #        model="HuggingFaceTB/SmolLM-1.7B-Instruct",
    #        task="text-generation",
    #        huggingface_pipeline_kwargs={"device_map": "auto",
    #                                     "model_kwargs": {"torch_dtype": torch.float16}},
    #        generation_kwargs={"max_new_tokens": 500, "temperature": 0.5, "do_sample": True})
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
    querying_pipeline.connect("embedder.embedding", "retriever.query_embedding") # Common connector
    querying_pipeline.connect("retriever", "answer_builder.documents")
    querying_pipeline.connect("retriever", "reader.documents") # Extractive QA Pipeline
    querying_pipeline.connect("retriever", "prompt_builder.documents") # Generative QA Pipeline
    querying_pipeline.connect("prompt_builder", "generator")
    querying_pipeline.connect("generator.replies", "answer_builder.replies")

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
                                   include_outputs_from = {"retriever", "prompt_builder", "reader", "generator", "answer_builder"} )

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

    st.write(":red[**complete->**]", answer)
    st.write(":red[**generator.replies->**]", answer["generator"]["replies"][0])
    st.write(":red[**reader.itemise->**] :blue[score=]", answer["reader"]["answers"][0].score, ":blue[, data=]", answer["reader"]["answers"][0].data)

    st.write(":green[**(1)retriever->**]", answer["retriever"])
    st.write(":green[**(1)retriever.object->**]", answer["retriever"]["documents"][0])
    st.write(":green[**(2)reader->**]", answer["reader"])
    st.write(":green[**(2)reader.object->**]", answer["reader"]["answers"][0])
    st.write(":green[**(3)generator->**]", answer["generator"])
    st.write(":green[**(3)generator.object->**]", answer["generator"]["replies"][0])
    st.write(":green[**(4)answer_builder->**]", answer["answer_builder"])
    st.write(":green[**(4)answer_builder.object->**]", answer["answer_builder"]["answers"][0])
    st.write(":green[**(5)prompt_builder->**]", answer["prompt_builder"])
    st.write(":green[**(5)prompt_builder.object->**]", answer["prompt_builder"]["prompt"])

    answer_builder_dict = (answer["answer_builder"]["answers"][0]).to_dict()
    answer_builder_object = (answer["answer_builder"]["answers"][0]).from_dict(answer_builder_dict)
    st.write(":blue[**answer_builder:0_dict->**]", answer_builder_dict)
    st.write(":blue[**answer_builder:0_object->**]", answer_builder_object)

def rag_qna_multiple():
    uploaded_files = st.file_uploader(":blue[**Choose multiple text/pdf files**]", type=['txt', 'pdf'], accept_multiple_files=True)
    if uploaded_files is not None:
        # To get file location for each file
        st.write("-->", uploaded_files)
        temp_dir = tf.mkdtemp()
        content_files = []
        files_list = []
        for upload_file in uploaded_files:
            temp_file = os.path.join(temp_dir, upload_file.name)
            content_files.append({"name": upload_file.name,
                                  "type": upload_file.type,
                                  "size": upload_file.size})
            files_list.append(temp_file)
            with open(mode="w+b", file=temp_file) as fn:
                fn.write(upload_file.getvalue())
                fn.close()
        if content_files is not None:
            logging.info(f"[ai] filelist information: {content_files}")
            st.write(f":green[dataset1:] {content_files}")
            st.write(f":green[dataset2:] {files_list}")
    else:
        st.markdown(":red[**Pls upload text/pdf files...**]")

if __name__ == '__main__':
    #if start_btn:
        tab01, tab02, tab03, tab04, tab05 = st.tabs(
            ["👻 RAG-Q&A-01", "👻 RAG-Q&A-02", "👻 RAG-Q&A-03", "👻 RAG-Q&A-multiple", "👻 RAG-Q&A-single"])
        with tab01:
            st.subheader("RAG-Q&A-01")
        with tab02:
            st.subheader("RAG-Q&A-02")
        with tab03:
            st.subheader("RAG-Q&A-03")
        with tab04:
            st.subheader("RAG-Q&A_multiple_content")
            rag_qna_multiple()
        with tab05:
            st.subheader("RAG-Q&A_single_content")
            rag_qna_single()
