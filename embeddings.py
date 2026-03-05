import pandas as pd
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.preprocessing import normalize


INPUT_FILE="Output/coursera_courses.csv"
OUTPUT_FILE="Output/top5_retrieved_courses.csv"

EMBED_MODEL="BAAI/bge-base-en-v1.5"
RERANK_MODEL="BAAI/bge-reranker-large"

TOP_K=5
FAISS_CANDIDATES=30


embedding_model=SentenceTransformer(EMBED_MODEL)
reranker=CrossEncoder(RERANK_MODEL)


def retrieve_courses(skill):

    print("\n--- RETRIEVING COURSES ---")

    df=pd.read_csv(INPUT_FILE)

    df=df.fillna("")

    df["combined"]=df["Name"]+" "+df["Skills"]

    documents=[
        "Represent this course for retrieval: "+text
        for text in df["combined"]
    ]

    embeddings=embedding_model.encode(documents)

    embeddings=normalize(embeddings)

    dimension=embeddings.shape[1]

    index=faiss.IndexFlatIP(dimension)

    index.add(embeddings)


    query_embedding=embedding_model.encode(
        ["Represent this sentence for searching relevant courses: "+skill]
    )

    query_embedding=normalize(query_embedding)

    scores,indices=index.search(query_embedding,FAISS_CANDIDATES)

    candidates=df.iloc[indices[0]].copy()

    pairs=[(skill,text) for text in candidates["combined"]]

    rerank_scores=reranker.predict(pairs)

    #candidates["score"]=rerank_scores

    candidates=candidates.sort_values(by="score",ascending=False)

    top=candidates.head(TOP_K)

    top.to_csv(OUTPUT_FILE,index=False)

    print("Saved:",OUTPUT_FILE)

    return top