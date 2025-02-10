import helper.config as cfg
import streamlit as st
import tempfile as tf
import pandas as pd
import os, logging, shutil, spacy

from pathlib import Path
from io import StringIO

st.set_page_config(page_title="Application #07", page_icon="🪷", layout="wide")
st.sidebar.title("🪷 Code-Testpad")
st.sidebar.markdown(
    """This is a testpad to generates baseline library. Enjoy!"""
)
st.sidebar.image(image="helper/eco-friendly.png", caption=None, use_column_width="always")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

st.write('Absolute path of file:  ', os.path.abspath(__file__))
st.write('Absolute directoryname: ', os.path.dirname(os.path.abspath(__file__)))
st.write('Current working path:   ', os.getcwd())

def test01():
    with st.container(border=True):
        st.subheader("Upload a file.")
        temp_dir = tf.mkdtemp()
        temp_images = {}
        temp_documents = {}
        with st.form("upload-documents", clear_on_submit=True, border=False):
            uploaded_file = st.file_uploader("Upload PDF or JPG files.", type=["txt","pdf"], accept_multiple_files=True)
            submitted = st.form_submit_button("Confirm Upload")
        if submitted and (uploaded_file is not None):
            for file in uploaded_file:
                st.write(f"File1: {file.getvalue()}")
                st.write(f"File2: {file.read()}")
                st.write(f"File name: {file}")
                temp_file = os.path.join(temp_dir, file.name)
                st.write(f"tempfile: ", temp_file)
                temp_documents[file.name] = temp_file
                with open(mode="w+b", file=temp_file) as fn:
                    fn.write(file.getvalue())
                    fn.close()
                with open(mode="r+b", file=temp_file) as fn:
                    st.write("111: ", fn.name)
                    st.write("222: ", fn.read())
                    fn.close()
            st.write("A: ", temp_images)
            st.write("B: ", temp_documents)
            shutil.rmtree(temp_dir)
            #os.remove(os.path.join(temp_dir, "*"))
            #os.rmdir(temp_dir)

def test02():
    uploaded_files = st.file_uploader(":blue[**Select (text/pdf) files**]", type=['txt', 'pdf'], accept_multiple_files=True)
    st.write("1-->", uploaded_files)
    if uploaded_files is not None:
        # To get file location for each file
        content_files = []
        files_list = []
        for upload_file in uploaded_files:
            content_files.append({"name": upload_file.name,
                                  "type": upload_file.type,
                                  "size": upload_file.size})
            files_list.append(Path(upload_file.name))
            upload_file.close()
        uploaded_files.clear()
        logging.info(f"[ai] uploaded_files: {uploaded_files}")
        logging.info(f"[ai] content_files: {content_files}")
        logging.info(f"[ai] files_list: {files_list}")
        st.write(f":green[dataset1:] {content_files}")
        st.write(f":green[dataset2:] {files_list}")
    else:
        st.markdown(":red[**Pls upload (text/pdf) files...**]")

def test03():
    uploaded_file = st.file_uploader(":blue[**Select (text/pdf) files**]", type=['txt', 'pdf'], accept_multiple_files=False)
    st.write("1-->", uploaded_file)
    if uploaded_file is not None:
        # To get file location for each file
        content_files = []
        files_list = []
        content_files.append({"name": uploaded_file.name,
                              "type": uploaded_file.type,
                              "size": uploaded_file.size})
        files_list.append(Path(uploaded_file.name))
        uploaded_file.close()
        logging.info(f"[ai] uploaded_files: {uploaded_file}")
        logging.info(f"[ai] content_files: {content_files}")
        logging.info(f"[ai] files_list: {files_list}")
        st.write(f":green[dataset1:] {content_files}")
        st.write(f":green[dataset2:] {files_list}")
    else:
        st.markdown(":red[**Pls upload (text/pdf) files...**]")

from haystack import Document

def test04():
    content_data = 0
    uploaded_file = st.file_uploader(":blue[**Choose a text file**]", type=['txt'], accept_multiple_files=False)
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        st.write(f":red[**bytes_data**]", bytes_data.decode('utf-8'))
        # To read file
        reads_data = uploaded_file.read()
        st.write(f":red[**reads_data**]", reads_data.decode('utf-8'))
        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        st.write(f":red[**stringio**]", stringio)
        # To read file as string:
        string_data = stringio.read()
        st.write(f":red[**string_data**]", string_data)
        # Can be used wherever a "file-like" object is accepted:
        #dataframe = pd.read_csv(uploaded_file, sep=None, delimiter=None, header=None, dtype=str)
        #st.write(f":red[**dataframe**]", dataframe)
        ###
        content_data = [Document(content=bytes_data, meta={"name": uploaded_file.name,
                                                            "type": uploaded_file.type,
                                                            "size": uploaded_file.size,
                                                            "url": uploaded_file._file_urls})]
        st.write(content_data)
        st.write(content_data[0])
        st.write(content_data[0].meta["name"])
        st.write(content_data[0].meta["type"])
        st.write(content_data[0].meta["size"])
    else:
        st.markdown(":red[**Pls upload a text file...**]")
    logging.info(f"[ai] uploaded_file: {uploaded_file}")
    logging.info(f"[ai] content_data: {content_data}")

from typing import List
from haystack import Document, Pipeline
from haystack.components.converters import TextFileToDocument, OutputAdapter
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from neo4j_haystack import Neo4jEmbeddingRetriever, Neo4jDocumentStore

def test05():
    converter = TextFileToDocument()
    documentx = converter.run(sources=["datasets/Sample_Text1.txt", "datasets/Sample_Text2.txt"])
    documents = documentx["documents"]
    st.write("documentx: ", documentx)
    st.write("documents: ", documents)
    #
    with st.expander("Shows details..."):
        document1 = documents[0]
        document_dict = document1.to_dict()
        document_object = document1.from_dict(document_dict)
        st.write("document:0_dict: ", document_dict)
        st.write("document:0_object: ", document_object)
        #
        #document_json = document.to_json()
        #document_object = document.from_json(document_json)
        #st.write("document:0_json: ", document_json)
        #st.write("document:0_object: ", document_object)
    #
    document_store = Neo4jDocumentStore(
        url="bolt://localhost:7687",
        username="neo4j",
        password="ne04j",
        database="neo4j",
        embedding_dim=384,
        embedding_field="embedding",
        index="document-embeddings",  # The name of the Vector Index in Neo4j
        node_label="Document",  # Providing a label to Neo4j nodes which store Documents
    )
    doc_written = document_store.write_documents(documents=documents, policy=DuplicatePolicy.OVERWRITE)
    st.write("Document written1: ", doc_written)

    document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    document_embedder.warm_up()
    documents_with_embeddings = document_embedder.run(documents)
    doc_written = document_store.write_documents(documents_with_embeddings.get("documents"))
    st.write("Document written2: ", doc_written)

    st.write(document_store.count_documents())

    pipeline = Pipeline()
    pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
    pipeline.add_component("retriever", Neo4jEmbeddingRetriever(document_store=document_store))
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    data = {
        "text_embedder": {"text": "What is 1mdb?"},
        "retriever": {"top_k": 5},
    }
    result = pipeline.run(data=data)
    documents: List[Document] = result["retriever"]["documents"]
    st.write(result)

from spacy import displacy
import spacy_streamlit as ss

def test06():
    text = "I am david lew living in puchong like to go travelling in thailand and work at google since 2020"
    nlp = spacy.load("en_core_web_sm")
    simi = spacy.load("en_core_web_lg")
    doc = nlp(text=text)
    logging.info(f"[ai] nlp-doc: {doc}")
    st.markdown(f"nlp-doc-> {doc}")
    #
    showme = displacy.render(doc, style="ent")
    st.html(showme)
    #
    ss.visualize_parser(doc)
    ss.visualize_ner(doc, labels=nlp.get_pipe("ner").labels)
    ss.visualize_textcat(doc)
    ss.visualize_similarity(simi, ("cat", "dog"))

def test07():
    filename = "datasets/Malaysia_Corruption_Reports.txt"
    with open(filename, "r", encoding="utf-8") as filehandler:
        dataset = filehandler.read()
    filehandler.close()

    text = "I am david lew living in puchong for 2 years like to go travelling in thailand to buy 200 bath shirts and work at google since 2020"
    models =  ["en_core_web_sm", "en_core_web_md"]
    visual = ["ner", "textcat"]
    #ss.visualize(models=models, default_text=dataset, visualizers=visual, show_logo=False)

    # For the non-transformer models, the ner component is independent, so you can disable everything else
    nlp1 = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    # In the transformer models, ner listens to the transformer component, so you can disable all components related tagging, parsing, and lemmatization
    # nlp2 = spacy.load("en_core_web_trf", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
    #
    nlp = nlp1
    doc = nlp(text=text)
    st.write(f"nlp-doc-> **{doc}**")
    st.write(f"nlp-doc-> **{doc.to_json()}**")
    ss.visualize_ner(doc, labels=nlp.get_pipe("ner").labels)
    st.html(displacy.render(doc, style="ent"))
    st.markdown(displacy.render(doc, style="ent"))
    st.stop()

def test08():
    converter = TextFileToDocument()
    documents = converter.run(sources=["datasets/Sample_Text1.txt", "datasets/Sample_Text2.txt"])
    st.write("documents-> ", documents, "total dosuments is ", len(documents["documents"]))
    st.write("documents[]-> ", [doc.content for doc in documents["documents"]])
    template = """{% for doc in documents["documents"] %}
                  'text': '{{doc.content}}',
                  'ents': [{{doc.meta['file_path']}},
                           {{doc.meta['file_path']}},
                           {{doc.meta['file_path']}}]
                  {% endfor %}
                  """
    adapter = OutputAdapter(template=template, output_type=dict)
    result = adapter.run(documents=documents)
    st.write("result-> ", result)

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

def test09():
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

test08()
