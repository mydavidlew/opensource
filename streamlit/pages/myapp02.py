import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Application #02", page_icon="🌼", layout="wide")
st.sidebar.title("🌼 Application #02")
st.sidebar.markdown(
    """This demo illustrates a combination of geospatial data visualisation, plotting and animation with 
    [**Streamlit**](https://docs.streamlit.io/develop/api-reference). We're generating a bunch of random numbers 
    in a loop for around 5 seconds. Enjoy!"""
)

DATE_COLUMN = 'dob'
DATA_URL = ('http://localhost/Duplicate_Patient_Demo_20240726.csv')

#DATE_COLUMN = 'date/time'
#DATA_URL = ('https://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache_data
def load_data(nrows):
    try:
        data = pd.read_csv(DATA_URL, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
        return data
    except Exception as e:
        st.write(f"Error: :red[**{e}**]")
        return None

def main():
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    # Load 10,000 rows of data into the dataframe.
    data = load_data(10000)
    # Notify the reader that the data was successfully loaded.
    data_load_state.text("Loading data...Done! (using st.cache_data)")

    if data:
        st.subheader('Raw data')
        st.write(data)

        st.subheader('Number of dob')
        hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
        st.bar_chart(hist_values)

        st.markdown('IDs')
        st.write(data['identification_no'])

if __name__ == '__main__':
    st.title("Read CSV from URL")
    main()