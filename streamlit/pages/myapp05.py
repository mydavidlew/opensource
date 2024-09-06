import streamlit as st
import pandas as pd
import os

from pathlib import Path
from datasets import load_dataset
from haystack import Pipeline, Document
from haystack.utils import ComponentDevice
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.converters import HTMLToDocument, MarkdownToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.readers import ExtractiveReader
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.builders import PromptBuilder
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.generators import OpenAIGenerator, HuggingFaceLocalGenerator
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.document_stores.in_memory import InMemoryDocumentStore
from tensorflow.python.ops.summary_ops_v2 import write
from tensorflow.tools.docs.doc_controls import header

st.set_page_config(page_title="Application #05", page_icon="🌹", layout="wide")
st.sidebar.title("🌹 Application #05")
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

filelist = ["datasets/Malaysia_Corruption_Reports.txt", "datasets/Malaysia_Corruption_1MDB.txt"]
filename = "datasets/Malaysia_Corruption_Reports.txt"
with open(filename, "r", encoding="utf-8") as filehandler:
    dataset = filehandler.read()
filehandler.close()

#dataset = load_dataset("text", data_files=filename, split="train")

documents = [Document(content=dataset, meta={"name": filename})]

#dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
#documents = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

st.write(dataset)

template = """Given these documents, answer the question.
              Documents:
              {% for doc in documents %}
                  {{ doc.content }}
              {% endfor %}
              Question: {{query}}
              Answer:"""
prompt_builder = PromptBuilder(template=template)

model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

document_store = InMemoryDocumentStore()
indexing_pipeline = Pipeline()
#indexing_pipeline.add_component(name="file_type_router", instance=FileTypeRouter(mime_types=["text/plain", "text/markdown", "application/pdf"]))
indexing_pipeline.add_component(name="plain_converter", instance=TextFileToDocument(encoding="utf-8"))
#indexing_pipeline.add_component(name="markdown_converter", instance=MarkdownToDocument())
#indexing_pipeline.add_component(name="pdf_converter", instance=PyPDFToDocument())
indexing_pipeline.add_component(name="joiner", instance=DocumentJoiner())
indexing_pipeline.add_component(name="cleaner", instance=DocumentCleaner())
indexing_pipeline.add_component(name="splitter", instance=DocumentSplitter(split_by="word", split_length=200, split_overlap=50))
indexing_pipeline.add_component(name="embedder", instance=SentenceTransformersDocumentEmbedder(model=model, progress_bar=True))
indexing_pipeline.add_component(name="writer", instance=DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP))
#indexing_pipeline.connect("file_type_router.text/plain", "ptext_converter.sources")
#indexing_pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
#indexing_pipeline.connect("file_type_router.application/pdf", "pdf_converter.sources")
indexing_pipeline.connect("plain_converter.documents", "joiner.documents")
#indexing_pipeline.connect("markdown_converter.documents", "joiner.documents")
#indexing_pipeline.connect("pdf_converter.documents", "joiner.documents")
indexing_pipeline.connect("joiner.documents", "cleaner.documents")
indexing_pipeline.connect("cleaner.documents", "splitter.documents")
indexing_pipeline.connect("splitter.documents", "embedder.documents")
indexing_pipeline.connect("embedder.documents", "writer.documents")

#indexing_pipeline.run({"file_type_router": {"sources": list(Path(output_dir).glob("**/*"))}})
indexing_pipeline.run({"plain_converter": {"sources": filelist}})
#indexing_pipeline.run(data={"joiner": {"documents": documents}})


reader_answer = ExtractiveReader(no_answer=False)
reader_answer.warm_up()
#generator = OpenAIGenerator()
generator = HuggingFaceLocalGenerator(
        model="HuggingFaceTB/SmolLM-1.7B-Instruct",
        huggingface_pipeline_kwargs={"device_map": "auto",
                                     "model_kwargs": {}},
        generation_kwargs={"max_new_tokens": 1000, "do_sample": True})
# Start the Generator
generator.warm_up()

retriever_store = InMemoryEmbeddingRetriever(document_store=document_store, top_k=10)
querying_pipeline = Pipeline()
querying_pipeline.add_component(name="embedder", instance=SentenceTransformersTextEmbedder(model=model, progress_bar=True))
querying_pipeline.add_component(name="retriever", instance=retriever_store)
querying_pipeline.add_component(name="reader", instance=reader_answer)
querying_pipeline.add_component(name="prompt_builder", instance=prompt_builder)
querying_pipeline.add_component(name="generator", instance=generator)
querying_pipeline.connect("embedder.embedding", "retriever.query_embedding")
querying_pipeline.connect("retriever.documents", "reader.documents")
querying_pipeline.connect("retriever.documents", "prompt_builder.documents")
querying_pipeline.connect("prompt_builder", "generator")


query = "What are the corruption cases in Malaysia?"
answer = querying_pipeline.run(data={"embedder": {"text": query},
                                     "retriever": {"top_k": 5},
                                     "reader": {"query": query, "top_k": 3},
                                     "generator": {"generation_kwargs": {"max_new_tokens": 500}}})
st.write(f":red[**a1->**]", answer)
st.write(f":red[**a2->**]", answer["reader"]["answers"][0])
st.write(f":red[**a3->**] :blue[score=]", answer["reader"]["answers"][0].score, f":blue[, data=]", answer["reader"]["answers"][0].data)
st.write(f":red[**a4->**]", answer["generator"]["replies"][0])
st.write(f":red[**b1->**]", reader_answer)
