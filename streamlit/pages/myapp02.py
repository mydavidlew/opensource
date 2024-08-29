import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Application #02", page_icon="🌍")
st.sidebar.title("🌍 Application #02")
st.sidebar.markdown(
    """This demo shows how to use
[`st.pydeck_chart`](https://docs.streamlit.io/develop/api-reference/charts/st.pydeck_chart)
to display geospatial data."""
)

DATE_COLUMN = 'dob'
DATA_URL = ('http://localhost/Duplicate_Patient_Demo_20240726.csv')

#DATE_COLUMN = 'date/time'
#DATA_URL = ('https://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Loading data...Done! (using st.cache_data)")

st.subheader('Raw data')
st.write(data)

st.subheader('Number of dob')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

st.markdown('IDs')
st.write(data['identification_no'])