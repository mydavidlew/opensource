import streamlit as st
import pandas as pd

st.set_page_config(page_title="Application #03", page_icon="🌻", layout="wide")
st.sidebar.title("🌻 Application #03")
st.sidebar.markdown(
    """This demo shows how to use
[`st.pydeck_chart`](https://docs.streamlit.io/develop/api-reference/charts/st.pydeck_chart)
to display geospatial data."""
)

# Specify the file paths
xlsx_file_path = '/home/superadmin/Workspace/Projects/Python/MyApp/datasets/Duplicate_Patient_Demo_20240726.xlsx'  # Replace with your .xlsx file path
csv_file_path = '/home/superadmin/Workspace/Projects/Python/MyApp/datasets/Duplicate_Patient_Demo_20240726a.csv'  # Replace with your desired .csv file path

# Function to convert xlsx to csv
def convert_xlsx_to_csv(xlsx_file_path, csv_file_path, sheet_name=3):
    # Read the Excel file
    df = pd.read_excel(xlsx_file_path, sheet_name=sheet_name)
    # Write the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)

# Function to read xlsx and filter distinct values in a specified range
def filter_distinct_values(xlsx_file_path, sheet_name=0, start_row=0, end_row=None, start_col=0, end_col=None):
    # Read the Excel file
    df = pd.read_excel(xlsx_file_path, sheet_name=sheet_name, header=None)
    # Select the specific range
    df_range = df.iloc[start_row:end_row, start_col:end_col]
    # Drop duplicates to filter distinct values
    distinct_values = df_range.drop_duplicates()
    return distinct_values

def doconvert():
    with st.spinner('Wait for it...'):
        # Convert the file
        convert_xlsx_to_csv(xlsx_file_path, csv_file_path)
    st.success("Done! --> " + xlsx_file_path)

def dofilter():
    # Specify the file path and the range
    sheet_name = 0  # Replace with your sheet name or index if needed
    start_row = 1  # Adjust these values to your desired range
    end_row = None  # None means until the last row
    start_col = 4  # Column index (0-based)
    end_col = 5  # None  # None means until the last column
    #
    with st.spinner('Wait for it...'):
        # Get the distinct values in the specified range
        distinct_values = filter_distinct_values(xlsx_file_path, sheet_name, start_row, end_row, start_col, end_col)
    st.success("Done! --> " + xlsx_file_path)
    # Print or use the distinct values
    st.write(distinct_values)

def doread():
    with st.spinner('Wait for it...'):
        df = pd.read_excel(xlsx_file_path, sheet_name=3, header=None, nrows=100)
    st.success("Done! --> " + xlsx_file_path)
    # Print or use the distinct values
    st.write(df)

with st.container():
    if st.button("Readfile", type="primary"):
        doread()

with st.container():
    if st.button("Convert", type="primary"):
        doconvert()

with st.container():
    if st.button("DistFilter", type="primary"):
        dofilter()