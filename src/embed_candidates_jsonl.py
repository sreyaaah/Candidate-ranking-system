import sqlite3
import numpy as np
import pickle
import torch
from sentence_transformers import SentenceTransformer
import config

def embed_candidates():
    torch.set_num_threads(4)
    print("Configured PyTorch to use 4 CPU threads.")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer(config.DENSE_MODEL_NAME, device=device)
    model.max_seq_length = config.MAX_SEQ_LENGTH
    
    # Quantize weights to INT8 only on CPU
    if device == "cpu":
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
    
    import os
    checkpoint_dir = "embeddings/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Process in batches of 500 candidates (2000 vectors)
    candidate_batch_size = 500
    all_embeddings_list = []
    all_names_list = []
    
    num_candidates = len(rows)
    print(f"Total candidates to embed: {num_candidates}")
    
    for i in range(0, num_candidates, candidate_batch_size):
        batch_num = i // candidate_batch_size
        checkpoint_file = os.path.join(checkpoint_dir, f"batch_{batch_num}.npz")
        
        if os.path.exists(checkpoint_file):
            print(f"Loading checkpoint for batch {batch_num}...")
            data = np.load(checkpoint_file, allow_pickle=True)
            chunk_embeddings = data['embeddings']
            chunk_names = list(data['names'])
            all_embeddings_list.append(chunk_embeddings)
            all_names_list.extend(chunk_names)
        else:
            print(f"Encoding batch {batch_num} (candidates {i} to {min(i + candidate_batch_size, num_candidates)})...")
            batch_rows = rows[i:i + candidate_batch_size]
            
            chunk_names = []
            chunk_queries = []
            
            for cand_id, summary, skills, experience, education, certs, full_text in batch_rows:
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
                    chunk_names.append(f"{cand_id}_{sec_name}")
                    chunk_queries.append("Represent this sentence for searching relevant passages: " + str(sec_text)[:1200])
            
            embeddings = model.encode(
                chunk_queries,
                batch_size=config.BATCH_SIZE,
                show_progress_bar=True,
                normalize_embeddings=True
            )
            chunk_embeddings = np.array(embeddings, dtype=np.float32)
            
            # Save batch checkpoint
            np.savez(checkpoint_file, embeddings=chunk_embeddings, names=np.array(chunk_names))
            
            all_embeddings_list.append(chunk_embeddings)
            all_names_list.extend(chunk_names)
            
    # Concatenate all batches
    print("Combining all checkpoints...")
    embeddings_np = np.vstack(all_embeddings_list)
    
    os.makedirs("embeddings", exist_ok=True)
    np.save("embeddings/resume_embeddings.npy", embeddings_np)
    with open("embeddings/resume_names.pkl", "wb") as f:
        pickle.dump(all_names_list, f)
        
    print(f"Saved {len(all_names_list)} section-level embeddings successfully.")

if __name__ == "__main__":
    embed_candidates()
