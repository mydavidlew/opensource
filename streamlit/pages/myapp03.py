import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Application #03", page_icon="🌻", layout="wide")
st.sidebar.title("🌻 Application #03")
st.sidebar.markdown(
    """This demo shows how to use
[`st.pydeck_chart`](https://docs.streamlit.io/develop/api-reference/charts/st.pydeck_chart)
to display geospatial data."""
)

# Specify the file paths
xlsx_file_path = '/home/superadmin/Workspace/datasets/Duplicate_Patient_Demo_20240726.xlsx'  # Replace with your .xlsx file path
csv_file_path = '/home/superadmin/Workspace/datasets/Duplicate_Patient_Demo_20240726a.csv'  # Replace with your desired .csv file path

# Function to get list of worksheet in xlsx
@st.cache_data
def readxls(xlsx_file_path):
    with st.spinner('Scanning excel, Wait for it...'):
        excel_file = pd.ExcelFile(xlsx_file_path)
    st.success("Done! --> " + xlsx_file_path.name)
    return excel_file.sheet_names

# Function to convert xlsx to csv
def convert_xlsx_to_csv(xlsx_file_path, csv_file_path, sheet_name=0):
    # Read the Excel file
    df = pd.read_excel(xlsx_file_path, sheet_name=sheet_name, header=0)
    # Write the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)

# Function to read xlsx and filter distinct values in a specified range
def filter_distinct_values(xlsx_file_path, sheet_name=0, start_row=0, end_row=None, start_col=0, end_col=None):
    # Read the Excel file
    df = pd.read_excel(xlsx_file_path, sheet_name=sheet_name, header=0)
    # Select the specific range
    df_range = df.iloc[start_row:end_row, start_col:end_col]
    # Drop duplicates to filter distinct values
    distinct_values = df_range.drop_duplicates()
    return distinct_values

def doconvert(xlsx_file_path, csv_file_path, sheet):
    with st.spinner('Wait for it...'):
        # Convert the file
        convert_xlsx_to_csv(xlsx_file_path, csv_file_path, sheet_name=sheet)
    st.success("Done! --> " + xlsx_file_path.name)

def dofilter(xlsx_file_path):
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
    st.success("Done! --> " + xlsx_file_path.name)
    # Print or use the distinct values
    st.write(distinct_values)

def domain(xlsx_file_path, csv_file_path):
    tab01, tab02, tab03, tab04, tab05 = st.tabs(["👻 Select worksheet", "👻 Read worksheet", "👻 Convert worksheet", "👻 Filter worksheet", "👻 Other worksheet"])
    sheet = 0
    with tab01:
        #if "df" not in st.session_state:
        #    st.session_state.df = readxls(xlsx_file_path)
        #items = st.dataframe(st.session_state.df, use_container_width=True, hide_index=False, on_select="rerun", selection_mode="single-row")
        df = readxls(xlsx_file_path)
        items = st.dataframe(df, use_container_width=True, hide_index=False, on_select="rerun", selection_mode="single-row")
        try:
            sheet = items.selection.rows[0]
        except IndexError:
            st.markdown(":red[***No***] :blue[*worksheet is selected!*]")
        st.write("Selected worksheet is: ", sheet)
        csv_file_path = csv_file_path + "_" + str(sheet) + ".csv"

    with tab02:
        with st.spinner('Reading excel, Wait for it...'):
            df = pd.read_excel(xlsx_file_path, sheet_name=sheet, header=0, nrows=100)
        st.success("Done! --> " + xlsx_file_path.name)
        st.table(df)

    with tab03:
        with st.spinner("Generating file 👉 " + csv_file_path):
            # Convert the file
            convert_xlsx_to_csv(xlsx_file_path, csv_file_path, sheet_name=sheet)
        st.success("Done! --> " + xlsx_file_path.name + " ➡️➡️➡️ " + csv_file_path)

    with tab04:
        st.write("d")

    with tab05:
        st.write("e")



uploaded_excel = st.file_uploader(":blue[**Choose a excel file**]", type=['xls','xlsx'], accept_multiple_files=False)
if uploaded_excel is not None:
    basefile, extension = os.path.splitext(uploaded_excel.name)
    downloaded_csv = "datasets/" + basefile
    domain(uploaded_excel, downloaded_csv)
else:
    st.markdown(":red[**Pls upload a excel file...**]")

#with st.container():
#    if st.button("Readfile", type="primary"):
#        doread()

#with st.container():
#    if st.button("Convert", type="primary"):
#        doconvert()

#with st.container():
#    if st.button("DistFilter", type="primary"):
#        dofilter()