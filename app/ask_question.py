# packages
from dotenv import load_dotenv
import chromadb
import openai
from anthropic import Anthropic
import os
from langchain_community.llms import Ollama
import pandas as pd

# env variables
load_dotenv()
openai_base_url = os.getenv('OPENAI_BASE_URL')
openai_api_key = os.getenv("LLM_TOKEN")

# local chroma db
CHROMA_DATA_PATH = "./chroma_data/"
collection_db="article_comments"

# Initialize am OpenAI client
client = openai.OpenAI(api_key=openai_api_key, base_url=openai_base_url)

# initialize a chroma client
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# define three small language models via Ollama (need ollama installed)
mistral = Ollama(model="mistral")
llama3 = Ollama(model="llama3:8b")
gemma = Ollama(model="gemma")

# helper function for prompt construction
def construct_prompt(docs: dict, question: str) -> str:
    # convert the docs into a numbered list of comments
    results_df = pd.DataFrame(docs['documents']).transpose()
    results_df.columns = ['Comment']
    results_df['ComNum'] = [str(i) for i in range(1, len(results_df) + 1)]
    results_df['Numbered Comments'] = results_df['ComNum'] + '. ' + results_df['Comment']

    # Collect the results in a context
    context = "\n\n".join([r for r in results_df['Numbered Comments']])

    # construct prompt
    prompt = f"""
        Answer the following question: {question}.  
        Refer only to the following numbered list of comments from NY Times readers when answering: {context}.
        Check each numbered comment very carefully and ignore it if it does not contain language that is a close match to the original question.
        Provide as much information as possible in the summary, subject to the conditions already given.
        Begin your answer with 'Based on the responses from selected NY Times readers', and try to give a sense of majority and minority opinions on the topic, but only if there is an identifiable majority opinion.
        If there is not enough information provided to give a summarized opinion, indicate that this is the case.
        """

    return prompt, context

# openai LLM RAG function
def ask_question_openai(question: str, client = client,
                        collection: chromadb.PersistentClient() = collection_db,
                        n_docs: int = 30, filters: dict = {}) -> str:
    # Find close documents in chromadb
    collection = chroma_client.get_collection(collection)
    results = collection.query(
        query_texts=[question],
        n_results=n_docs,
        where=filters
    )

    prompt = construct_prompt(results, question)
    sent_prompt = prompt[0]
    sent_context = prompt[1]

    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": sent_prompt,
        }],
        model="gpt-4",
    )

    return chat_completion.choices[0].message.content, sent_context

# initialize an anthropic client
anthropic_client = Anthropic(
    base_url=os.getenv("ANTHROPIC_BASE_URL"),
    api_key=os.getenv("LLM_TOKEN")
)

# Claude 3 function
def ask_question_anthropic(question:str, collection: chromadb.PersistentClient() = collection_db,
                           client = anthropic_client, n_docs: int = 30, filters: dict = {}) -> str:
    # Find close documents in chromadb
    collection = chroma_client.get_collection(collection)
    results = collection.query(
        query_texts=[question],
        n_results=n_docs,
        where=filters
    )

    prompt = construct_prompt(results, question)
    sent_prompt = prompt[0]
    sent_context = prompt[1]

    res = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=2000,
        messages=[
            {"role": "user", "content": sent_prompt},
        ],
    )

    return res.content[0].text, sent_context


# local SLM rag function
def ask_question_local_slm(question: str, llm: Ollama() = llama3,
                           collection: chromadb.PersistentClient() = collection_db, n_docs: int = 30,
                           filters: dict = {}) -> str:
    # Find close documents in chromadb
    collection = chroma_client.get_collection(collection)
    results = collection.query(
        query_texts=[question],
        n_results=n_docs,
        where=filters
    )

    prompt = construct_prompt(results, question)
    sent_prompt = prompt[0]
    sent_context = prompt[1]

    return llm.invoke(sent_prompt), sent_context