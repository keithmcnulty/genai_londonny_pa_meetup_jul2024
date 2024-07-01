# initiates and populates vector database with comments from pickle file created in kaggle_set_to_pickle.py

from langchain_community.document_loaders import DataFrameLoader
import pandas as pd
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.batch_utils import create_batches
import uuid

# load comments df and filter out short or empty comments
comments = pd.read_pickle("comments.pickle")
comments = comments[comments['commentBody'].notnull()]
comments['COUNT'] = [np.char.count(comment, ' ') for comment in comments['commentBody']]
longer_comments = comments[comments['COUNT'] >= 40]

# load into langchain document format
loader = DataFrameLoader(longer_comments, page_content_column="commentBody")
docs = loader.load()

# set up the ChromaDB
CHROMA_DATA_PATH = "./chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "article_comments"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# uncomment in case docs have already been written
# client.delete_collection(COLLECTION_NAME)

# enable the DB using Cosine Similarity as the distance metric
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

# create document collection in ChromaDB
collection = client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"},
)

# chromadb has a batch size limit for writing
# create batches with year as metadata, and random UUID
batches = create_batches(
    api=client,
    ids=[f"{uuid.uuid4()}" for i in range(len(docs))],
    documents=[doc.page_content for doc in docs],
    metadatas=[{'year': docs[k].metadata['year']} for k in range(len(docs))]
)

# write batches to chromaDB - over 2M comments n batches of ~40K
# for each comment, chromaDB will store the comment, year, and embedding
# one time write - this will take a while
# if you are impatient you can cut down the number of comments (eg choose a specific month)
for batch in batches:
    print(f"Adding batch of size {len(batch[0])}")
    collection.add(ids=batch[0],
                   documents=batch[3],
                   metadatas=batch[2])

# test data
# results = collection.query(
#     query_texts=["Opinions on US foreign policy towards North Korea "],
#     n_results=10,
#     include=['documents', 'metadatas', 'embeddings'],
#     where={'year': 2017}
# )
#
# results
