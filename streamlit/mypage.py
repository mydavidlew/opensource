import streamlit as st

def main():
    # builds the sidebar menu
    with st.sidebar:
        st.page_link('mypage.py', label='Main Application Page', icon='🏠')
        st.page_link('pages/myapp01.py', label='MyApp01 Checker', icon='🛡️')
        st.page_link('pages/myapp02.py', label='MyApp02 Checker', icon='🌍')

    #st.title(f'🔥 Welcome to MyPages!')
    st.write("# 🏠 Welcome to MyPages! 👋")
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


if __name__ == '__main__':
    st.set_page_config(page_title="Hello to MyPage", page_icon="👋")
    st.sidebar.success("Select an apps above.")
    main()
