import sqlite3
import numpy as np
import pickle
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DB_PATH = "data/candidates.db"

def embed_candidates():
    # Use BAAI/bge-base-en-v1.5 (or BGE-M3 later) which is 100% free and local
    print("Loading local embedding model (BAAI/bge-base-en-v1.5)...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    
    # If a GPU is available, SentenceTransformer will use it automatically.
    # Otherwise, it will use your CPU.
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM candidates")
    total_candidates = cursor.fetchone()[0]
    
    cursor.execute("SELECT id, full_text FROM candidates")
    
    embeddings = []
    resume_names = []
    
    batch_size = 256 # Larger batch size for local processing
    batch_ids = []
    batch_texts = []
    
    print(f"Generating embeddings for {total_candidates} candidates locally...")
    print("This may take 1-2 hours depending on your CPU, but it is 100% free and unlimited!")
    
    for row in tqdm(cursor, total=total_candidates):
        cand_id, text = row
        
        # BGE models use this prefix for retrieving relevant passages
        query = "Represent this sentence for searching relevant passages: " + text[:4000]
        
        batch_ids.append(cand_id)
        batch_texts.append(query)
        
        if len(batch_texts) >= batch_size:
            # Encode locally
            batch_embeddings = model.encode(
                batch_texts, 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings.extend(batch_embeddings)
            resume_names.extend(batch_ids)
            
            batch_texts = []
            batch_ids = []
            
    # Process remaining
    if batch_texts:
        batch_embeddings = model.encode(
            batch_texts, 
            normalize_embeddings=True,
            show_progress_bar=False
        )
        embeddings.extend(batch_embeddings)
        resume_names.extend(batch_ids)

    conn.close()
    
    embeddings_np = np.array(embeddings, dtype=np.float32)
    
    # Save files
    np.save("embeddings/resume_embeddings.npy", embeddings_np)
    with open("embeddings/resume_names.pkl", "wb") as f:
        pickle.dump(resume_names, f)
        
    print(f"Saved {len(embeddings)} embeddings successfully.")

if __name__ == "__main__":
    embed_candidates()
