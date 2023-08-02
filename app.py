import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

from model import *

# Set API Key
import os
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

print(os.environ['OPENAI_API_KEY'])

# Set session state
def clear_submit():
    st.session_state["submit"] = False

st.set_page_config(page_title="HireFire - An LLM-powered hiring assistant", page_icon=":bulb:", layout='wide')
st.header("HireFire")

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

# Layout of input/response containers
input_container = st.container()
context_container = st.container()
colored_header(label='', description='', color_name='blue-30')
button = st.button("Submit")
response_container = st.container()

# Flag
isPDF = False

# User input
## Function for taking user provided PDF as input
def get_file(key):
    uploaded_files = st.file_uploader(f"Upload your {key}", type='pdf', key=key, on_change=clear_submit, accept_multiple_files=True)
    return uploaded_files

## Applying the user input box
with input_container:
    files = get_file('resumes')

def get_text(prompt):
    input_text = st.text_area(prompt, "", key="input",  on_change=clear_submit)
    return input_text

## Applying the user context box
with context_container:
    context_text = get_text('Ask me anything about the applicants!')

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(fileobj, context_text):

    text = get_text_from_pdf(fileobj)

    summary = custom_prompt_summary(text, context_text, chain_type='map_reduce')
    recommendation = get_recommendation(text, context_text, chain_type='map_reduce')
    
    return summary, recommendation

## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if button or st.session_state.get("submit"):

        if not files:
            st.error("Please upload at least one resume!")
        elif not context_text:
            st.error("Please enter a question!")
        else:
            st.session_state["submit"] = True

            with st.spinner('Wait for it...'):
                summary, recommendation = generate_response(user_input, context_text)
                output = f'''
                    Summary:
                        {summary}
                    
                    Recommendations:
                        {recommendation}
                '''
            st.success('Done!')

            message(output)
