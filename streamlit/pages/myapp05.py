import streamlit as st
import pandas as pd
import os

from datasets import load_dataset
from haystack import Pipeline, Document
from haystack.utils import ComponentDevice
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.converters import HTMLToDocument, TextFileToDocument
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.readers import ExtractiveReader
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


dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
documents = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

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
indexing_pipeline.add_component(name="converter", instance=TextFileToDocument(encoding="utf-8"))
indexing_pipeline.add_component(name="cleaner", instance=DocumentCleaner())
indexing_pipeline.add_component(name="splitter", instance=DocumentSplitter(split_by="word", split_length=200, split_overlap=10)))
indexing_pipeline.add_component(name="embedder", instance=SentenceTransformersDocumentEmbedder(model=model))
indexing_pipeline.add_component(name="writer", instance=DocumentWriter(document_store=document_store))
indexing_pipeline.connect("converter.documents", "cleaner.documents")
indexing_pipeline.connect("cleaner.documents", "splitter.documents")
indexing_pipeline.connect("splitter.documents", "embedder.documents")
indexing_pipeline.connect("embedder.documents", "writer.documents")
indexing_pipeline.run(data={"converter": {"documents": documents}})


reader_answer = ExtractiveReader(no_answer=False)
reader_answer.warm_up()
generator = OpenAIGenerator()

retriever_store = InMemoryEmbeddingRetriever(document_store=document_store, top_k=5)
querying_pipeline = Pipeline()
querying_pipeline.add_component(name="embedder", instance=SentenceTransformersTextEmbedder(model=model))
querying_pipeline.add_component(name="retriever", instance=retriever_store)
querying_pipeline.add_component(name="reader", instance=reader_answer)
querying_pipeline.add_component(name="prompt_builder", instance=prompt_builder)
querying_pipeline.add_component(name="llm", instance=generator)
querying_pipeline.connect("embedder.embedding", "retriever.query_embedding")
querying_pipeline.connect("retriever.documents", "reader.documents")
querying_pipeline.connect("retriever.documents", "prompt_builder.documents")
querying_pipeline.connect("prompt_builder", "llm")


query = "Who was Pliny the Elder?"
answer = querying_pipeline.run(
    data={"embedder": {"text": query}, "retriever": {"top_k": 3}, "reader": {"query": query, "top_k": 2}}
)

st.write(answer)
