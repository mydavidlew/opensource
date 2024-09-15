import helper.config as cfg
import streamlit as st
import os, logging, shutil
from pathlib import Path
import tempfile as tf

st.set_page_config(page_title="Application #07", page_icon="🪷", layout="wide")
st.sidebar.title("🪷 Code-Testpad")
st.sidebar.markdown(
    """This is a testpad to generates baseline library. Enjoy!"""
)

st.write('Absolute path of file:  ', os.path.abspath(__file__))
st.write('Absolute directoryname: ', os.path.dirname(os.path.abspath(__file__)))
st.write('Current working path:   ', os.getcwd())

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

uploaded_files = st.file_uploader(":blue[**Select (text/pdf) files**]", type=['txt', 'pdf'], accept_multiple_files=True)
if uploaded_files is not None:
    # To get file location for each file
    st.write("1-->", uploaded_files)
    content_files = []
    files_list = []
    for upload_file in uploaded_files:
        content_files.append({"name": upload_file.name,
                              "type": upload_file.type,
                              "size": upload_file.size})
        files_list.append(Path(upload_file.name))
        st.write("2-->", files_list)
    if content_files is not None:
        logging.info(f"[ai] filelist information: {content_files}")
        st.write(f":green[dataset1:] {content_files}")
        st.write(f":green[dataset2:] {files_list}")
else:
    st.markdown(":red[**Pls upload (text/pdf) files...**]")