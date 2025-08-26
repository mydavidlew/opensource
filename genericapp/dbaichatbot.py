# Chat With Your Database Using Natural Language: Build a Local AI-Powered SQL Chat App
#
# In today’s data-driven world, accessing and analyzing information stored in databases should be straightforward for everyone, not just SQL experts. 
# What if non-technical team members could simply ask questions in plain English and get answers from your database? In this article, I’ll walk you 
# through building exactly that — a conversational interface for your MySQL database using Python, Streamlit, and Large Language Models with the 
# powerful Reflection Pattern.
#
# https://medium.com/@vishwajeetv2003/building-a-natural-language-database-query-tool-chat-with-your-mysql-database-using-the-reflection-1e5e3a2149d6
#
import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from utils.mysqlconnector import MySqlConnector  # Import our custom MySQL connector class
from utils.psqlconnector import PostgreSQLConnector  # Import our custom PostgreSQL connector class

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client for LLM API access
client = Groq(
    #api_key=os.environ.get("GROQ_API_KEY"),  # Get API key from environment variables
    api_key="dlgsk_MQyJyBApNw8oIGPGWAmRWGdyb3FYOHuUSNps68TA81N7D5BMdTMy"
)

def get_llm_response(prompt):
    """
    Send a prompt to the Groq LLM API and get a response
    Utilizes Llama 3.3 70B model for high-quality text generation
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def validate_query(query, schema_info):
    """
    Part of the Reflection Pattern: This function validates if the generated SQL query is correct
    by sending it to the LLM to check against the database schema
    """
    prompt = f"""
    You are a SQL validator. Check this query for errors:
    Query: "{query}"
    Database Schema: {schema_info}
    Check for syntax, missing tables/columns, etc.
    Respond "VALID" or explain the error.
    """
    return get_llm_response(prompt)

def iterative_query_generation(user_input, schema_info, max_retries=3):
    """
    Reflection Pattern implementation: Generates SQL from natural language with validation
    1. Generate initial SQL query
    2. Validate the query
    3. If errors exist, use feedback to improve and try again
    4. Repeat up to max_retries times
    """
    query_writer_prompt = f"""
    You are a SQL expert. Write a MySQL query to answer the user's request below
    User Request : "{user_input}".
    Use ONLY these tables/columns: {schema_info}.
    Respond ONLY with the SQL query. Do NOT include explanations or any markdowns
    """
    # Get initial query from LLM
    generated_query = get_llm_response(query_writer_prompt)

    # Reflection loop: Validate and refine
    for _ in range(max_retries):
        validation_result = validate_query(generated_query, schema_info)
        if "VALID" in validation_result:
            return generated_query  # Return query if valid
        else:
            # Add error feedback to prompt and retry
            query_writer_prompt += f"\nPrevious error: {validation_result}\nRewrite the query:"
            generated_query = get_llm_response(query_writer_prompt)

    return "Failed to generate valid query after retries."

# Add custom CSS styling for better UI appearance
st.markdown("""
    <style>
        .connection-form {
            max-width: 400px;
            padding: 20px;
            background-color: #f0f2f6;
            border-radius: 10px;
            margin: 20px auto;
        }
        .connection-form h2 {
            color: #1f77b4;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state variables to maintain state between re-runs
if "db_connected" not in st.session_state:
    st.session_state.db_connected = False  # Track database connection status
if "db_info" not in st.session_state:
    st.session_state.db_info = {}  # Store database connection details
if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history storage

# Database connection interface - show only if not already connected
if not st.session_state.db_connected:
    with st.container():
        st.markdown('<div class="connection-form">', unsafe_allow_html=True)
        st.header("Database Connection Setup 🔌")

        # Connection form for database credentials
        with st.form("db_connection"):
            host = st.text_input("Host", placeholder="localhost")
            user = st.text_input("User", placeholder="root")
            password = st.text_input("Password", type="password")
            database = st.text_input("Database", placeholder="employees")
            submitted = st.form_submit_button("Connect")

            if submitted:
                with st.spinner("Connecting to database..."):
                    # Attempt to establish connection
                    connection = MySqlConnector(host, user, password, database)
                    conn = connection.get_connection()
                    if conn is not None:
                        # If successful, update session state and refresh
                        st.session_state.db_connected = True
                        st.session_state.db_info = {
                            "host": host,
                            "user": user,
                            "password": password,
                            "database": database,
                        }
                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
else:
    # Main chat interface - only shown after successful database connection
    st.title(f"Chatting with {st.session_state.db_info['database']} 💬")

    # Initialize MySQL connector with saved credentials
    mysql = MySqlConnector(
        host=st.session_state.db_info['host'],
        user=st.session_state.db_info['user'],
        password=st.session_state.db_info['password'],
        database=st.session_state.db_info['database']
    )
    conn = mysql.get_connection()  # Create connection object

    # Get database schema information for query generation
    schema_info = mysql.get_basic_info()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input field
    if prompt := st.chat_input("Ask your database question..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate SQL query using Reflection Pattern
        final_query = iterative_query_generation(prompt, schema_info)

        # Execute the query against the database
        response = mysql.execute_pd_query(final_query)

        # Display assistant response with query results
        with st.chat_message("assistant"):
            st.write(response)  # Display results as a pandas DataFrame
        st.session_state.messages.append({"role": "assistant", "content": response})
