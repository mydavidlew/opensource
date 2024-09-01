import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import random
import time
import os

from haystack import Pipeline
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import HTMLToDocument
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator

st.set_page_config(page_title="Application #04", page_icon="🪻", layout="wide")
st.sidebar.title("🪻 Application #04")
st.sidebar.markdown(
    """This demo shows how to use
[`st.pydeck_chart`](https://docs.streamlit.io/develop/api-reference/charts/st.pydeck_chart)
to display geospatial data."""
)


os.environ["OPENAI_API_KEY"] = Application_key

# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

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
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                stream = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    stream=True,
                )
                response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.write(f"Error: :red[**{e}**]")

def main():
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
        st.write(f"Error: :red[**{e}**]")

if __name__ == '__main__':
    st.title("Simple AI chatbot")
    #main()
    chatgpt()