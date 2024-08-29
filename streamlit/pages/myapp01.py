import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Application #01", page_icon="🌸", layout="wide")
st.sidebar.title("🌸 Application #01")
st.sidebar.markdown(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

st.write("Hello world")
df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

option = st.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected: ', option

st.write(df)
st.dataframe(df.style.highlight_max(axis=0))
st.table(df)

x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * 2)

map_data = pd.DataFrame(
    np.random.randn(x * 2, 2) / [5, 5] + [3.10, 101.70],
    columns=['lat', 'lon'])

st.map(map_data)
