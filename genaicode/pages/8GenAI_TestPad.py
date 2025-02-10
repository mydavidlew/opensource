import helper.config as cfg
import streamlit as st
import pandas as pd
import os, logging, torch, spacy
import tempfile as tf

from openai import OpenAI
from pathlib import Path
from spacy import displacy
from datasets import load_dataset
from haystack import Pipeline
from haystack import Document
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
from haystack.components.extractors import NamedEntityExtractor, NamedEntityAnnotation
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.document_stores.in_memory import InMemoryDocumentStore
from tensorflow.python.ops.summary_ops_v2 import write
from tensorflow.tools.docs.doc_controls import header

st.set_page_config(page_title="Application #08", page_icon="🌹", layout="wide")
st.sidebar.title("🌹 RAG-Testpad")
st.sidebar.markdown(
    """This demo illustrates the capability of GenAI using Haystack and [**Streamlit**](https://docs.streamlit.io/develop/api-reference)
    frameworks. This testpad generates a series of AI tools based on the concept of RAG. Enjoy!"""
)
#start_btn = st.sidebar.button(f"Click to **Start**", type="primary")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logging.getLogger("haystack").setLevel(logging.INFO)

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = cfg.openai_key
if "HF_API_TOKEN" not in os.environ:
    os.environ["HF_API_TOKEN"] = cfg.dlreadtoken_key
    os.environ["HF_TOKEN"] = cfg.dlreadtoken_key

def rag_qna_single():
# [1 load the documents or files
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
# 1]

# [2 creation of indexing_pipeline
    model_embedder = "sentence-transformers/all-mpnet-base-v2" # better & larger model than below
    #model_embedder = "sentence-transformers/all-MiniLM-L12-v2"
    #model_embedder = "sentence-transformers/all-MiniLM-L6-v2"
    #model_embedder = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    #device_embedder = ComponentDevice.from_str("cuda:0") # train using accelerate GPU
    device_embedder = None

    extractor_hf = NamedEntityExtractor(backend="hugging_face", model="dslim/bert-base-NER")
    extractor_sp = NamedEntityExtractor(backend="spacy", model="en_core_web_sm")
    EntityExtractor = extractor_sp

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
    indexing_pipeline.add_component(name="extractor", instance=EntityExtractor)
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
    indexing_pipeline.connect("joiner", "extractor")
    #
    #indexing_results = indexing_pipeline.run({"file_type_router": {"sources": foldername}})
    #indexing_results = indexing_pipeline.run(data={"joiner": {"documents": documents}})
    indexing_results = indexing_pipeline.run(data={"plain_converter": {"sources": filelist}},
                                             include_outputs_from={"embedder", "writer", "extractor"})
    with st.expander("indexing_results"):
        st.write(":blue[**indexing_results->**]", indexing_results)
        st.write(":blue[**documents.splitter#->**]", indexing_results["writer"]["documents_written"])
    #
    # NER of the uploaded documents
    nerdocuments = indexing_results["extractor"]["documents"]
    annotations = [NamedEntityExtractor.get_stored_annotations(doc) for doc in nerdocuments]
    with st.expander("extractor_results"):
        st.write(":blue[**extractor.documents->**]", nerdocuments, "consist of: ", len(nerdocuments), " documents")
        st.write(":blue[**extractor.annotations->**]", annotations)
    with st.expander(f"extractor_visualise: {nerdocuments[0].meta['file_path']}"):
        nlp_object = {} # Dictionary in keys & values pair
        nlp_entity = [] # List of objects
        entity_list = nerdocuments[0].meta["named_entities"]
        for i in range(0, len(entity_list)):
            nlp_entity.append({'start': entity_list[i].start,
                               'end': entity_list[i].end,
                               'label': entity_list[i].entity})
        nlp_object.update({"text": nerdocuments[0].content})
        nlp_object.update({"ents": nlp_entity})
        st.html(displacy.render(nlp_object, manual=True, style="ent"))
# 2]

# [3 creation of querying_pipeline
    template = """Given these documents, answer the question.
        Context:
        {% for doc in documents %}
        Document: {{ loop.index }} - File: {{ doc.meta['file_path'] }}
        {{ doc.content }}
        {% endfor %}
        Question: {{query}}
        Answer: """
    prompt_builder = PromptBuilder(template=template)
    #
    reader_answer = ExtractiveReader(no_answer=False)
    reader_answer.warm_up()
    #
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
# 3]

# [4 ask a question to the RAG model
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
    querying_results = querying_pipeline.run(data=data,
                                             include_outputs_from={"retriever", "prompt_builder", "reader", "generator", "answer_builder"})
    answer = querying_results
    with st.expander("querying_results"):
        st.write(":red[**querying_results->**]", answer)
# 4]

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
    #pipe["prompt_builder"]["prompt"]

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

    with st.expander("answer_builder"):
        answer_builder_dict = (answer["answer_builder"]["answers"][0]).to_dict()
        answer_builder_object = (answer["answer_builder"]["answers"][0]).from_dict(answer_builder_dict)
        st.write(":blue[**answer_builder:0_dict->**]", answer_builder_dict)
        st.write(":blue[**answer_builder:0_object->**]", answer_builder_object)

def rag_qna_multiple():
    uploaded_files = st.file_uploader(":blue[**Choose multiple text/pdf files**]", type=['txt', 'pdf'], accept_multiple_files=True)
    if uploaded_files is not None:
        # To get file location for each file
        st.write(uploaded_files)
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
        if files_list is not None:
            logging.info(f"[ai] filelist information: {files_list}")
            st.write(f":green[dataset1:] {content_files}")
            st.write(f":green[dataset2:] {files_list}")
    else:
        st.markdown(":red[**Pls upload text/pdf files...**]")

def entity_extractor():
    # Get the data from a specific file only
    filename = "datasets/Malaysia_Corruption_Reports.txt"
    with open(filename, "r", encoding="utf-8") as filehandler:
        dataset = filehandler.read()
    filehandler.close()
    # dataset = load_dataset("text", data_files=filename, split="train")
    documents = [Document(content=dataset, meta={"name": filename})]
    logging.info(f"[ai] documents information: {documents}")
    #
    extractor_hf = NamedEntityExtractor(backend="hugging_face", model="dslim/bert-base-NER")
    extractor_sp = NamedEntityExtractor(backend="spacy", model="en_core_web_sm")
    extractor = extractor_hf
    extractor.warm_up()
    results = extractor.run(documents=documents)
    annotations = [NamedEntityExtractor.get_stored_annotations(doc) for doc in results["documents"]]
    #
    st.write("documents: ", documents)
    st.write("result: ", results)
    st.write("result.documents: ", results["documents"])
    st.write("result.annotations: ", annotations)

def chatgpt():
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "user", "content": "How can i help you?"}]
    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(message["content"])

    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            try:
                stream = client.chat.completions.create(model=st.session_state["openai_model"],
                                                        messages=[{"role": m["role"], "content": m["content"]}
                                                                  for m in st.session_state.messages],
                                                        stream=True)
                response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                logging.info(f"Error: :red[**{e}**]")
                st.write(f"Error: :red[**{e}**]")

def simplechat():
    fetcher = LinkContentFetcher()
    converter = HTMLToDocument()
    prompt_template = """According to the contents of this website:
        {% for document in documents %}
            {{document.content}}
        {% endfor %}
        Question: {{query}}
        Answer:"""
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
        data = {"fetcher": {"urls": ["https://haystack.deepset.ai/overview/quick-start"]},
                "prompt": {"query": "Which components do I need for a RAG pipeline?"}}
        result = pipeline.run(data=data, include_outputs_from={"converter", "prompt", "llm"})
        st.write("Results: ", result)
        st.write("Replies: ", result["llm"]["replies"][0])
    except Exception as e:
        logging.info(f"Error: :red[**{e}**]")
        st.write(f"Error: :red[**{e}**]")

def get_generative_answer(query_pipeline, query):
  results = query_pipeline.run({
      "text_embedder": {"text": query},
      "prompt_builder": {"query": query}
    })
  answer = results["generator"]["replies"][0]
  return answer

def ragchat():
    uploaded_file = st.file_uploader(":blue[**Choose a excel file**]", type=['xls', 'xlsx'], accept_multiple_files=False)
    if uploaded_file is not None:
        content_file = pd.read_fwf(uploaded_file)
        content_data = [Document(content=content_file)]
        st.write(content_data)
        #
        # In memory document store
        document_store = InMemoryDocumentStore()
        #🚅 Components
        #- splitter: DocumentSplitter
        #- embedder: SentenceTransformersDocumentEmbedder
        #- writer: DocumentWriter
        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="word", split_length=200))
        indexing_pipeline.add_component("embedder",
            SentenceTransformersDocumentEmbedder(
                model="Snowflake/snowflake-arctic-embed-l", # good embedding model: https://huggingface.co/Snowflake/snowflake-arctic-embed-l
                device=None,                                # load the model on GPU = ComponentDevice.from_str("cuda:0")
            ))
        indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
        #🛤️ Connections
        #- splitter.documents -> embedder.documents(List[Document])
        #- embedder.documents -> writer.documents(List[Document])
        indexing_pipeline.connect("splitter", "embedder")
        indexing_pipeline.connect("embedder", "writer")
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
        generator = HuggingFaceLocalGenerator(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            huggingface_pipeline_kwargs={"device_map": "auto",
                                         "model_kwargs": {"load_in_4bit": True,
                                                          "bnb_4bit_use_double_quant": True,
                                                          "bnb_4bit_quant_type": "nf4",
                                                          "bnb_4bit_compute_dtype": torch.bfloat16}},
            generation_kwargs={"max_new_tokens": 500})
        generator.warm_up()
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

if __name__ == '__main__':
    #if start_btn:
        tab01, tab02, tab03, tab04, tab05 = st.tabs(
            ["👻 RAG-Q&A-01", "👻 RAG_Chatbot", "👻 Entity_Extractor", "👻 ChatGPT/SimpleChat", "👻 RAG+Q&A_Content"])
        with tab01:
            st.subheader("RAG-Q&A-01")
        with tab02:
            st.subheader("RAG_Chatbot")
            ragchat()
        with tab03:
            st.subheader("Entity_Extractor")
            entity_extractor()
        with tab04:
            st.subheader("ChatGPT/SimpleChat")
            #chatgpt()
            simplechat()
        with tab05:
            st.subheader("RAG+Q&A_Content")
            #rag_qna_single()
            rag_qna_multiple()
