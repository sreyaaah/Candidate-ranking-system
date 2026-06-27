import sqlite3
import numpy as np
import pickle
import torch
from sentence_transformers import SentenceTransformer
import config

def embed_candidates():
    torch.set_num_threads(4)
    print("Configured PyTorch to use 4 CPU threads.")
    
    print(f"Loading local embedding model ({config.DENSE_MODEL_NAME})...")
    model = SentenceTransformer(config.DENSE_MODEL_NAME, local_files_only=True)
    model.max_seq_length = config.MAX_SEQ_LENGTH
    
    # Quantize weights to INT8 to accelerate CPU execution and reduce memory overhead
    try:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("Model dynamic INT8 quantization enabled successfully.")
    except Exception as e:
        print(f"Dynamic quantization fallback: {e}")
    
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    # Select structured columns directly to avoid fragile string splitting
    cursor.execute("SELECT id, summary, skills, experience, education, certifications, full_text FROM candidates")
    rows = cursor.fetchall()
    conn.close()
    
    resume_names = []
    queries = []
    
    print("Packing candidate profiles into section vectors...")
    for cand_id, summary, skills, experience, education, certs, full_text in rows:
        summary_text = str(summary or "")
        skills_text = str(skills or "")
        experience_text = str(experience or "")
        education_certs_text = f"Education: {education or ''}. Certifications: {certs or ''}."
        
        # Section mapping
        sections = {
            "summary": summary_text if len(summary_text) > 10 else str(full_text)[:300],
            "skills": skills_text if len(skills_text) > 10 else str(full_text)[:300],
            "experience": experience_text if len(experience_text) > 10 else str(full_text)[:500],
            "education": education_certs_text if len(education_certs_text) > 10 else str(full_text)[-300:]
        }
        
        for sec_name, sec_text in sections.items():
            resume_names.append(f"{cand_id}_{sec_name}")
            queries.append("Represent this sentence for searching relevant passages: " + str(sec_text)[:1200])
            
    print(f"Generating embeddings for {len(queries)} section-level vectors locally...")
    
    embeddings = model.encode(
        queries,
        batch_size=config.BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    
    embeddings_np = np.array(embeddings, dtype=np.float32)
    
    import os
    os.makedirs("embeddings", exist_ok=True)
    np.save("embeddings/resume_embeddings.npy", embeddings_np)
    with open("embeddings/resume_names.pkl", "wb") as f:
        pickle.dump(resume_names, f)
        
    print(f"Saved {len(embeddings)} section-level embeddings successfully.")

if __name__ == "__main__":
    embed_candidates()
