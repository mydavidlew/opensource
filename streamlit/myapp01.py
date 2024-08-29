import streamlit as st
import pandas as pd
import numpy as np

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
