import pandas as pd
import sqlite3
import os
import sys
import numpy as np
from sentence_transformers import CrossEncoder

# Ensure src is in the path to import search_hybrid
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from search_hybrid import get_hybrid_scores
from docx import Document
from extract_skills import extract_required_skills

def generate_training_data():
    print("Loading Job Description...")
    doc = Document("data/jobs/job_description.docx")
    jd_text = "\n".join([para.text for para in doc.paragraphs])
    
    required_skills = extract_required_skills(jd_text)
    print(f"Dynamically extracted JD skills: {required_skills}")
    
    top_n = 4000
    hybrid_results = get_hybrid_scores(jd_text, top_n=top_n)
    
    print("\nFetching features from database...")
    candidate_ids = [res[0] for res in hybrid_results]
    
    conn = sqlite3.connect("data/candidates.db")
    
    placeholders = ",".join(["?"] * len(candidate_ids))
    query = f"""
        SELECT 
            c.id, c.skills, c.full_text, f.*
        FROM candidates c
        LEFT JOIN candidate_features f ON c.id = f.candidate_id
        WHERE c.id IN ({placeholders})
    """
    df = pd.read_sql_query(query, conn, params=candidate_ids)
    # Remove duplicate candidate_id column if present from f.*
    df = df.loc[:, ~df.columns.duplicated()]
    conn.close()
    
    print("\nLoading CrossEncoder for Knowledge Distillation (Local Inference)...")
    ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    # Give the model the title/context, plus the explicit requirements section
    jd_trunc = jd_text[:300] + "\n...[Requirements]...\n" + jd_text[3800:5200]
    
    dataset = []
    
    print("Preparing data and running teacher model inference...")
    for cand_id, score in hybrid_results:
        row = df[df["id"] == cand_id]
        if row.empty:
            continue
            
        row = row.iloc[0]
        text = str(row["skills"]).lower()
        matched = sum(1 for skill in required_skills if skill in text)
        skill_score = matched / len(required_skills) if required_skills else 0.0
        
        full_text = str(row["full_text"])[:1000]
        
        # Build features dict dynamically from row
        feat_dict = {
            "candidate_id": cand_id,
            "full_text": full_text,
            "rrf_score": score,
            "skill_score": skill_score,
        }
        
        # Copy over all numeric features from SQL
        exclude_cols = {'id', 'skills', 'full_text', 'candidate_id'}
        for col in df.columns:
            if col not in exclude_cols:
                val = row[col]
                # Convert nulls to 0 or appropriate default
                if pd.isna(val):
                    val = 3.0 if col == 'education_tier' else 0.0
                feat_dict[col] = val
                
        dataset.append(feat_dict)
        
    final_df = pd.DataFrame(dataset)
    
    # Run Cross-Encoder on all 4000 to get gold-standard labels
    ce_pairs = [(jd_trunc, txt) for txt in final_df["full_text"]]
    ce_scores = ce_model.predict(ce_pairs)
    
    # Normalize CE scores between 0 and 1
    norm_ce_scores = 1 / (1 + np.exp(-ce_scores))
    final_df["teacher_score"] = norm_ce_scores
    
    # Force honeypots to 0 teacher score
    final_df.loc[final_df["honeypot_flag"] == 1, "teacher_score"] = 0.0
    
    # Create relevance labels based on percentiles of the teacher score
    q95 = final_df["teacher_score"].quantile(0.95)
    q80 = final_df["teacher_score"].quantile(0.80)
    q50 = final_df["teacher_score"].quantile(0.50)
    
    def assign_label(score):
        if score >= q95: return 3
        if score >= q80: return 2
        if score >= q50: return 1
        return 0
        
    final_df["relevance"] = final_df["teacher_score"].apply(assign_label)
    
    # Save training dataset
    os.makedirs("data", exist_ok=True)
    final_df.to_csv("data/training_data.csv", index=False)
    print("Saved data/training_data.csv successfully!")
    print(final_df["relevance"].value_counts())

if __name__ == "__main__":
    generate_training_data()
