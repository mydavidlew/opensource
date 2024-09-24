import helper.config as cfg
import streamlit as st
import tempfile as tf
import pandas as pd
import os, logging, shutil, spacy

from pathlib import Path
from io import StringIO
from haystack import Document
from haystack.components.converters import TextFileToDocument, OutputAdapter

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

def test05():
    converter = TextFileToDocument()
    documents = converter.run(sources=["datasets/Sample_Text1.txt", "datasets/Sample_Text2.txt"])
    document = documents["documents"][0]
    st.write("documents: ", documents)
    st.write("document:0: ", document)
    #
    document_dict = document.to_dict()
    document_object = document.from_dict(document_dict)
    st.write("document:0_dict: ", document_dict)
    st.write("document:0_object: ", document_object)
    #
    document_json = document.to_json()
    document_object = document.from_json(document_json)
    st.write("document:0_json: ", document_json)
    st.write("document:0_object: ", document_object)

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

test08()
