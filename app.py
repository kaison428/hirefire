import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

from model import *

# Set API Key ----------------------------------------------------------------
import os
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
os.environ['ZAPIER_NLA_API_KEY'] = st.secrets['ZAPIER_NLA_API_KEY']

# Set session state
def clear_submit():
    st.session_state["submit"] = False

st.set_page_config(page_title="HireFire - An LLM-powered hiring assistant", page_icon=":bulb:", layout='wide')
st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.header("HireFire - Understand your resume database")

# Sidebar contents
with st.sidebar:
    st.title(':bulb: HireFire')
    st.markdown('''
    ## About
    This app is an LLM-powered hiring assistant built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/en/latest/)
    
    ''')

    add_vertical_space(5)
    st.write('Made by HYPE AI :book:')

# Layout of input/response containers ----------------------------------------------------------------
input_container = st.container()
context_container = st.container()
colored_header(label='', description='', color_name='blue-30')
process_button = st.button("Process")
chat_container = st.container()

# User input ------------------------------------------------------------------
## Function for taking user provided PDF as input
def get_file(key):
    uploaded_files = st.file_uploader(f"Upload your {key}", type='pdf', key=key, on_change=clear_submit, accept_multiple_files=True)
    return uploaded_files

## Applying the user input box
with input_container:
    files = get_file('resumes')

# Process Button ------------------------------------------------------------------
def get_agent_from_data(files):
    # 1. Read and process data
    resumes = []
    for file in files:
        resumes.append(get_text_from_pdf(file))

    # 2. Build an agent from the database
    agent = get_agent(resumes)
    # Note: this agent stays in the app space and will be fed to the backend for post processing
    return agent

if process_button:
    if not files:
        st.error("Please upload at least one document!")
    else:
        with st.spinner('Processing...'):
            st.session_state["agent"] = get_agent_from_data(files)

        st.success('Done!')
        st.session_state["submit"] = True

# Chatbot GUI ----------------------------------------------------------------
with chat_container:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.session_state.get("submit"):
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask me anything about the resumes!"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                agent = st.session_state["agent"]
                full_response = agent.run(prompt)
                message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
