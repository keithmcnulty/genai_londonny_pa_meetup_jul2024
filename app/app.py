# packages
import streamlit as st
import chromadb
import openai
import os
from dotenv import load_dotenv

# load objects and functions from ask_question.py
from ask_question import *

# streamlit page config
st.set_page_config(page_title='AskNYT', layout='wide')

# additional options sidebar (model, number of docs from chroma, year filter)
with st.sidebar.expander(" 🛠️ Settings ", expanded=False):
    model = st.selectbox(label = "Model", options = ['GPT-4 (LLM)', 'Llama3:9b (SLM)', 'Gemma:8b (SLM)', 'Mistral (SLM)'])
    n_documents = st.number_input('Number of comments to be selected', min_value=20, max_value=100)
    year = st.selectbox(label = "Year", options = ['All', '2017', '2018'])

if year == 'All':
        filter = {}
elif year == '2017':
        filter = {'year': 2017}
else:
        filter = {'year': 2018}

# app title layout
st.title("AskNYT 🤖")
st.markdown(
        ''' 
        > :black[**Your informative friend:**  *powered by NYT Readers*]
        ''')

# app input
with st.form(key='my_form'):
        input = st.text_area(label="Enter your question for the readers of the NYT:")
        submit_button = st.form_submit_button(label='Submit')

# output to be executed on submit
if submit_button:
        with st.spinner('Generating response...'):
                if model == 'GPT-4 (LLM)':
                        output = ask_question_openai(input, client = client, n_docs=n_documents, filters = filter)
                elif model == 'Llama3:9b (SLM)':
                        output = ask_question_local_slm(input, llm = llama3, n_docs=n_documents, filters = filter)
                elif model == 'Mistral (SLM)':
                        output = ask_question_local_slm(input, llm = mistral, n_docs=n_documents, filters = filter)
                else:
                        output = ask_question_local_slm(input, llm = gemma, n_docs=n_documents, filters = filter)

                st.write("**Summary opinion:**\n\n" + output[0] + '\n\n**The comments used to generate this summary were as follows:**\n\n' + output[1])


