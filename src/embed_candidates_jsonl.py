import sqlite3
import numpy as np
import pickle
import torch
from sentence_transformers import SentenceTransformer

DB_PATH = "data/candidates.db"

def embed_candidates():
    # Set PyTorch threads to prevent memory thrashing and core contention on CPU
    torch.set_num_threads(4)
    print("Configured PyTorch to use 4 CPU threads.")
    
    print("Loading local embedding model (BAAI/bge-base-en-v1.5)...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    model.max_seq_length = 384 # Set seq length slightly lower to save massive CPU cache memory
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, full_text FROM candidates")
    rows = cursor.fetchall()
    conn.close()
    
    resume_names = [row[0] for row in rows]
    # Slice to 1500 characters: fits full 300+ tokens, capturing Summary, Skills, Certs, and Recent Roles
    queries = ["Represent this sentence for searching relevant passages: " + str(row[1])[:1500] for row in rows]
    
    print(f"Generating embeddings for {len(queries)} candidates locally (batch size 16 to fit CPU cache)...")
    
    embeddings = model.encode(
        queries,
        batch_size=16, # Small batch size prevents CPU cache bottlenecks and OOMs
        show_progress_bar=True,
        normalize_embeddings=True
    )
    
    embeddings_np = np.array(embeddings, dtype=np.float32)
    
    # Save files
    import os
    os.makedirs("embeddings", exist_ok=True)
    np.save("embeddings/resume_embeddings.npy", embeddings_np)
    with open("embeddings/resume_names.pkl", "wb") as f:
        pickle.dump(resume_names, f)
        
    print(f"Saved {len(embeddings)} embeddings successfully.")

if __name__ == "__main__":
    embed_candidates()
