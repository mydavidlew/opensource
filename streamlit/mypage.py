import streamlit as st

# builds the sidebar menu
st.set_page_config(page_title="Hello to MyPage", page_icon="👋", layout="wide")
st.sidebar.title(f'👉 Homepage! 👈')
st.sidebar.success("Select a task above ☝️")
st.sidebar.markdown(
    """
    Dedicated built for :green[**Artificial Intelligent**] and :green[**Graph Data Science**] projects.
"""
)
st.sidebar.image(image="helper/eco-friendly.png", caption=None, use_column_width="always")
st.logo(image="helper/eco-friendly.png", icon_image="helper/eco-friendly.png")

def main():
    # content of the page
    st.markdown(
        """
        **Streamlit** is an open-source app framework built specifically for
        Machine Learning and Data Science projects.
        **👈 Select a demo from the sidebar** to see some examples
        of what Streamlit can do!
        ### Want to learn more?
        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
            forums](https://discuss.streamlit.io)
        ### See more complex demos
        - Use a neural net to [analyze the Udacity Self-driving Car Image
            Dataset](https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
    )

def submain():
    # builds the sidebar menu
    with st.sidebar:
        st.page_link('mypage.py', label='Main Application', icon='🏠')
        st.page_link('pages/myapp01.py', label='RAG-Single', icon='🌸️')
        st.page_link('pages/myapp02.py', label='RAG-Multiple', icon='🌾')

if __name__ == '__main__':
    st.title("🏠 Welcome to MyPages! 👋")
    main()
    submain()
