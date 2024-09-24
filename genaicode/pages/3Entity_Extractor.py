import streamlit as st
import spacy as sp
import spacy_streamlit as ss
import os, logging

from spacy import displacy
from haystack import Document
from spacy.util import minify_html

st.set_page_config(page_title="Application #03", page_icon="🤖", layout="wide")
st.sidebar.title("🤖 Entity Extraction")
st.sidebar.markdown(
    """This Named Entity Recognition (NER) illustrates the used of spaCy NLP framework, for Entity
    Extraction on any text documents. Enjoy!"""
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger("haystack").setLevel(logging.INFO)

def upload_file():
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
            st.session_state.content_data = [Document(content=bytes_data.decode('utf-8'), meta={"name": uploaded_file.name, "type": uploaded_file.type, "size": uploaded_file.size, "url": uploaded_file._file_urls})]
        if st.session_state.content_data is not None:
            st.write(f"[ File: :orange[**{st.session_state.content_data[0].meta['name']}**] --- Type: :orange[**{st.session_state.content_data[0].meta['type']}**] --- Size: :orange[**{st.session_state.content_data[0].meta['size']}**] bytes ]")
    logging.info(f"[ai] uploaded_file: {uploaded_file}")
    logging.info(f"[ai] content_data: {st.session_state.content_data}")
    return st.session_state.content_data

def ner_file(models, visual, selected):
    content_data = upload_file()
    if content_data is not None:
        text = content_data[0].content
        #text = "I am david lew living in puchong like to go travelling in thailand and work at google since 2020"
        #st.write("document-> ", content_data)
        #st.write("content-> ", text)

        # For the non-transformer models, the ner component is independent, so you can disable everything else
        # nlp = sp.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
        # In the transformer models, ner listens to the transformer component, so you can disable all components related tagging, parsing, and lemmatization
        # nlp2 = spacy.load("en_core_web_trf", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
        #
        show_all = False
        if show_all is True:
            ss.visualize(models=models, default_text=text, visualizers=visual, show_visualizer_select=False, show_logo=False)
        else:
            if text != "":
                nlp = sp.load(selected)
                doc = nlp(text=text)
                ss.visualize_ner(doc=doc, labels=nlp.get_pipe("ner").labels, key="nerv1a")
                with st.expander("JSON Document"):
                    st.write("json-doc->", doc.to_json())
                with st.expander("Dictionary Document"):
                    st.write("dict-doc->", doc.to_dict())
                with st.expander("Parser Visualisation"):
                    ss.visualize_parser(doc=doc, key="nerv1b")
                with st.expander("spaCy Visualisation"):
                    htmldoc = displacy.render(docs=doc, style="ent", minify=True)
                    st.html(htmldoc)
                    st.divider()
                    st.markdown(f":blue[**html-doc->**] {htmldoc}")

def ner_text(models, visual, selected):
    text = st.text_area(label="Enter any text here...", height=300)
    submit = st.button(label="Submit")
    if text != "" or (text != "" and submit is True):
        nlp = sp.load(selected)
        doc = nlp(text=text)
        ss.visualize_ner(doc=doc, labels=nlp.get_pipe("ner").labels, key="nerv2a")
        with st.expander("JSON Document"):
            st.write("json-doc->", doc.to_json())
        with st.expander("Dictionary Document"):
            st.write("dict-doc->", doc.to_dict())
        with st.expander("Parser Visualisation"):
            ss.visualize_parser(doc=doc, key="nerv2b")
        with st.expander("spaCy Visualisation"):
            htmldoc = displacy.render(docs=doc, style="ent", minify=True)
            st.html(htmldoc)
            st.divider()
            st.markdown(f":blue[**html-doc->**] {htmldoc}")

if __name__ == '__main__':
    # st.title("Entity Extraction")
    reset_btn = st.sidebar.button(f"Click to **Reset**", type="primary", use_container_width=True)
    if reset_btn is True:
        st.session_state.content_data = None
        st.session_state.clear()
        st.cache_data.clear()
        st.cache_resource.clear()
    models = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]
    visual = ["parser", "ner", "textcat", "similarity", "tokens"]
    selected = st.sidebar.selectbox(label="NLP Models", options=models, index=0)
    st.sidebar.image(image="helper/eco-friendly.png", caption=None, use_column_width="always")
    tab01, tab02 = st.tabs(["👻 NER File", "👻 NER Text"])
    with tab01:
        ner_file(models, visual, selected)
    with tab02:
        ner_text(models, visual, selected)
