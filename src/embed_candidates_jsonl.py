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
    model.max_seq_length = 384 # Optimal seq length for CPU cache performance
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, full_text FROM candidates")
    rows = cursor.fetchall()
    conn.close()
    
    resume_names = []
    queries = []
    
    print("Parsing candidate profiles into structured sections...")
    for cand_id, text in rows:
        text = str(text)
        
        # Safe extraction of structured sections from candidate full_text
        summary = ""
        skills = ""
        experience = ""
        education = ""
        
        try:
            if "Profile Summary: " in text:
                summary = text.split("Profile Summary: ")[1].split(". Core Competencies: ")[0]
            if "Core Competencies: " in text:
                skills = text.split("Core Competencies: ")[1].split(" Experience: ")[0]
            if " Experience: " in text:
                experience = text.split(" Experience: ")[1].split(". Education: ")[0]
            if "Education: " in text:
                education = text.split("Education: ")[1]
        except Exception:
            # Fallback if parsing fails due to malformed text
            summary = text[:500]
            skills = text[:500]
            experience = text[:1000]
            education = text[-500:]
            
        # Add 4 sections per candidate
        sections = {
            "summary": summary if summary else text[:300],
            "skills": skills if skills else text[:300],
            "experience": experience if experience else text[:500],
            "education": education if education else text[-300:]
        }
        
        for sec_name, sec_text in sections.items():
            resume_names.append(f"{cand_id}_{sec_name}")
            # Prep query prefix for retrieval passage search
            queries.append("Represent this sentence for searching relevant passages: " + str(sec_text)[:1200])
            
    print(f"Generating embeddings for {len(queries)} section-level vectors locally...")
    
    # Encode all 20,000 section vectors directly with smart batching
    embeddings = model.encode(
        queries,
        batch_size=16, # fits L3 CPU cache perfectly
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
        
    print(f"Saved {len(embeddings)} section-level embeddings successfully.")

if __name__ == "__main__":
    embed_candidates()
