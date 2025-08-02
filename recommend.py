import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import gzip


def load_embeddings():

    with gzip.open('hybrid_embeddings.pkl.gz', 'rb') as f1:
        hybrid_embeddings = pickle.load(f1)

    with gzip.open("user_hybrid_embeddings.pkl.gz", "rb") as f2:
        user_hybrid_vectors = pickle.load(f2)
        
    return hybrid_embeddings, user_hybrid_vectors


def generate_recommendations(user_id, hybrid_embeddings, user_hybrid_vectors, top_n=5):
    if user_id not in user_hybrid_vectors:
        return None

    user_vec = user_hybrid_vectors[user_id]
    all_other_ids = list(hybrid_embeddings.keys())

    candidate_vecs = np.array([hybrid_embeddings[oid] for oid in all_other_ids])
    sim_scores = cosine_similarity([user_vec], candidate_vecs)[0]

    rec_df = pd.DataFrame({
        "othermemberid": all_other_ids,
        "similarity": sim_scores
    }).sort_values(by="similarity", ascending=False).head(top_n)

    return rec_df.to_dict(orient="records")