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
            c.id, c.skills, c.full_text,
            f.yoe, f.ai_years, f.job_hopping_index, f.github_score, 
            f.education_tier, f.notice_period_days, f.recruiter_response_rate,
            f.company_fit_score, f.career_trajectory_score, f.behavioral_score,
            f.location_match, f.honeypot_flag
        FROM candidates c
        LEFT JOIN candidate_features f ON c.id = f.candidate_id
        WHERE c.id IN ({placeholders})
    """
    df = pd.read_sql_query(query, conn, params=candidate_ids)
    conn.close()
    
    print("\nLoading CrossEncoder for Knowledge Distillation (Local Inference)...")
    ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    jd_trunc = jd_text[:1000]
    
    dataset = []
    
    print("Preparing data and running teacher model inference...")
    for cand_id, score in hybrid_results:
        row = df[df["id"] == cand_id]
        if row.empty:
            continue
            
        row = row.iloc[0]
        text = str(row["skills"]).lower()
        matched = sum(1 for skill in required_skills if skill in text)
        skill_score = matched / len(required_skills)
        
        full_text = str(row["full_text"])[:1000]
        
        dataset.append({
            "candidate_id": cand_id,
            "full_text": full_text,
            "rrf_score": score,
            "skill_score": skill_score,
            "yoe": float(row["yoe"]),
            "ai_years": float(row["ai_years"]),
            "job_hopping_index": float(row["job_hopping_index"]),
            "github_score": float(row["github_score"]),
            "education_tier": int(row["education_tier"]),
            "notice_period": int(row["notice_period_days"]),
            "company_fit_score": float(row["company_fit_score"]),
            "career_trajectory_score": float(row["career_trajectory_score"]),
            "behavioral_score": float(row["behavioral_score"]),
            "location_match": int(row["location_match"]),
            "honeypot_flag": int(row["honeypot_flag"])
        })
        
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
