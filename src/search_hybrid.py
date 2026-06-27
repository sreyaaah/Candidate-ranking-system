import numpy as np
import faiss
import pickle
import sqlite3
import re
from rank_bm25 import BM25Okapi

def tokenize(text):
    if not text:
        return []
    return re.findall(r"(?i)\b[a-z0-9_+#.]+\b", text.lower())

def get_hybrid_scores(jd_text, k_rrf=60, top_n=2000):
    # 1. Load dense resources
    print("Loading FAISS index and JD embedding...")
    jd_embedding = np.load("embeddings/jd_embedding.npy")
    index = faiss.read_index("embeddings/resume.index")
    with open("embeddings/resume_names.pkl", "rb") as f:
        dense_resume_names = pickle.load(f)

    # Dense Search on Section Index
    k_dense = min(top_n * 4, len(dense_resume_names))
    dense_scores, dense_indices = index.search(jd_embedding.reshape(1, -1), k_dense)
    
    # Late Fusion Aggregation: map section hits back to parent candidates, keeping the highest rank (Min Rank)
    seen_candidates = {}
    dense_candidate_list = []
    
    for idx in dense_indices[0]:
        sec_id = dense_resume_names[idx]
        # Split at the last underscore to get parent cand_id
        parts = sec_id.rsplit("_", 1)
        cand_id = parts[0]
        
        if cand_id not in seen_candidates:
            seen_candidates[cand_id] = len(dense_candidate_list)
            dense_candidate_list.append(cand_id)
            
    # 2. Load sparse resources
    print("Loading BM25 index...")
    with open("embeddings/bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open("embeddings/bm25_names.pkl", "rb") as f:
        sparse_resume_names = pickle.load(f)
        
    # Sparse Search
    tokenized_query = tokenize(jd_text)
    sparse_scores = bm25.get_scores(tokenized_query)
    
    # Map candidate IDs to their sparse rank
    sparse_scores_with_ids = list(zip(sparse_resume_names, sparse_scores))
    sparse_scores_with_ids.sort(key=lambda x: x[1], reverse=True)
    
    # RRF calculation
    rrf_scores = {}
    
    # Populate dense ranks using late-fused candidate positions
    for rank, cand_id in enumerate(dense_candidate_list[:top_n]):
        if cand_id not in rrf_scores:
            rrf_scores[cand_id] = 0.0
        rrf_scores[cand_id] += 1.0 / (k_rrf + rank + 1)
        
    # Populate sparse ranks
    for rank, (cand_id, score) in enumerate(sparse_scores_with_ids):
        if cand_id not in rrf_scores:
            rrf_scores[cand_id] = 0.0
        rrf_scores[cand_id] += 1.0 / (k_rrf + rank + 1)
        
    # Sort by hybrid RRF score
    hybrid_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return hybrid_ranked[:top_n]

if __name__ == "__main__":
    from docx import Document
    print("Loading Job Description...")
    doc = Document("data/jobs/job_description.docx")
    jd_text = "\n".join([para.text for para in doc.paragraphs])
    
    hybrid_results = get_hybrid_scores(jd_text, top_n=20)
    print("\nTop 20 Hybrid RRF Candidates:\n")
    for i, (cand_id, score) in enumerate(hybrid_results):
        print(f"{i+1}. {cand_id} -> RRF Score: {score:.4f}")
