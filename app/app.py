# packages
import streamlit as st
import chromadb
import openai
import os
from dotenv import load_dotenv
from ask_question import ask_question_openai, ask_question_local_slm, llama3, gemma, mistral, client, chroma_client

# load env variables
load_dotenv()

# streamlit page config
st.set_page_config(page_title='AskNYT', layout='wide')

# gpt-4 model
MODEL_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = 'gpt-4'

# API key via environment variable
API_KEY = os.getenv("LLM_TOKEN")

CHROMA_DATA_PATH = "./chroma_data/"
collection_db="article_comments"

# Initialize a client
client = openai.OpenAI(api_key=API_KEY, base_url=MODEL_URL)

chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# additional options sidebar
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

# Set up the app layout
st.title("AskNYT 🤖")
st.markdown(
        ''' 
        > :black[**Your informative friend:**  *powered by NYT Readers*]
        ''')

with st.form(key='my_form'):
        input = st.text_area(label="Enter your question for the readers of the NYT:")
        submit_button = st.form_submit_button(label='Submit')

if submit_button:
        with st.spinner('Generating response...'):
                if model == 'GPT-4 (LLM)':
                        output = ask_question_openai(input, n_docs=n_documents, filters = filter)
                elif model == 'Llama3:9b (SLM)':
                        output = ask_question_local_slm(input, llm = llama3, n_docs=n_documents, filters = filter)
                elif model == 'Mistral (SLM)':
                        output = ask_question_local_slm(input, llm = mistral, n_docs=n_documents, filters = filter)
                else:
                        output = ask_question_local_slm(input, llm = gemma, n_docs=n_documents, filters = filter)

                st.write(output)


