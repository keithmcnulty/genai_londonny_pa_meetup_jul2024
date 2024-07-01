from dotenv import load_dotenv
import chromadb
import openai
import os
from langchain_community.llms import Ollama

load_dotenv()
openai_base_url = os.getenv('OPENAI_BASE_URL')
openai_api_key = os.getenv("LLM_TOKEN")
CHROMA_DATA_PATH = "./chroma_data/"
collection_db="article_comments"

#Initialize a client
client = openai.OpenAI(api_key=openai_api_key, base_url=openai_base_url)

chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

mistral = Ollama(model="mistral")
llama3 = Ollama(model="llama3:8b")
gemma = Ollama(model="gemma")


def ask_question_openai(question: str, collection: chromadb.PersistentClient() = collection_db, n_docs: int = 30,
                 filters: dict = {}) -> str:
    # Find close documents in chromadb
    collection = chroma_client.get_collection(collection)
    results = collection.query(
        query_texts=[question],
        n_results=n_docs,
        where=filters
    )

    # Collect the results in a context
    context = "\n".join([r for r in results['documents'][0]])

    prompt = f"""
    Answer the following question: {question}.  
    Refer only to the following information when answering: {context}.
    Provide at least six paragraphs in summary if it is possible given the information provided.
    Begin your answer with 'Based on the responses from selected NY Times readers', and try to give a sense of majority and minority opinions on the topic, but only if there is an identifiable majority opinion.
    If there is not enough information provided to give a summarized opinion, indicate that this is the case.
    """

    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": prompt,
        }],
        model="gpt-4",
    )

    return chat_completion.choices[0].message.content

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

    # Collect the results in a context
    context = "\n".join([r for r in results['documents'][0]])

    prompt = f"""
    Answer the following question: {question}.  
    Refer only to the following information when answering: {context}.
    Provide at least six paragraphs in summary if it is possible given the information provided.
    Begin your answer with 'Based on the responses from selected NY Times readers', and try to give a sense of majority and minority opinions on the topic, but only if there is an identifiable majority opinion.
    If there is not enough information provided to give a summarized opinion, indicate that this is the case.
    """

    return llm.invoke(prompt)