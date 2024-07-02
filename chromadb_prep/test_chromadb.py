import pandas as pd
import numpy as np
import chromadb

# get chromadb collection we made earlier
CHROMA_DATA_PATH = "./chroma_data/"
COLLECTION_NAME = "article_comments"
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = client.get_collection(COLLECTION_NAME)

# test data
results = collection.query(
    query_texts=["What are readers opinions on US foreign policy towards North Korea?"],
    n_results=10,
    include=['documents', 'metadatas', 'embeddings'],
    where={'year': 2017}
)

# convert results to nice dataframe
results_df = pd.Dataframe(results['documents']).transpose()
results_df.columns = ['Comment']
results.insert(loc=0, column='CommentNumber', value=np.arange(len(results_df))+1)
results[]