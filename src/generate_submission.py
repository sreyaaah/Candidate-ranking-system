import numpy as np
import pandas as pd
import sqlite3
import os
import sys
import xgboost as xgb
import csv

# Ensure src is in the path to import search_hybrid
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from search_hybrid import get_hybrid_scores
from docx import Document

required_skills = [
    "python",
    "react",
    "node.js",
    "docker",
    "sql"
]

def generate_reasoning(row):
    """
    Stage 7 - Explainability
    Generates a 1-2 sentence evidence-based reasoning string.
    """
    yoe = float(row.get("yoe", 0.0))
    tier = int(row.get("education_tier", 3))
    notice = int(row.get("notice_period", 0))
    github = float(row.get("github_score", 0.0))
    
    tier_str = "Tier 1" if tier == 1 else "Tier 2" if tier == 2 else "Tier 3"
    
    reason = f"Candidate selected due to {yoe:.1f} years of experience and {tier_str} educational background. "
    if github > 70:
        reason += f"Demonstrates exceptionally strong technical engagement (GitHub score: {github:.1f}). "
    if notice <= 30:
        reason += "Highly favorable notice period allows for immediate onboarding."
    elif notice > 60:
        reason += f"Notice period of {notice} days is a minor risk but offset by strong semantic match."
    else:
        reason += "Solid overall fit for the required technical stack."
        
    return reason.strip()

def main():
    print("Loading Job Description...")
    doc = Document("data/jobs/job_description.docx")
    jd_text = "\n".join([para.text for para in doc.paragraphs])
    
    # 1. Stage 1 & 2: Hybrid Retrieval (Fetch extra candidates so we can filter honeypots)
    top_n = 2000
    hybrid_results = get_hybrid_scores(jd_text, top_n=top_n)
    
    print("\nFetching features from database...")
    candidate_ids = [res[0] for res in hybrid_results]
    
    # 2. Stage 3: Feature Engineering lookups
    conn = sqlite3.connect("data/candidates.db")
    cursor = conn.cursor()
    
    placeholders = ",".join(["?"] * len(candidate_ids))
    query = f"""
        SELECT 
            c.id, c.skills, 
            f.yoe, f.ai_years, f.job_hopping_index, f.github_score, 
            f.education_tier, f.notice_period_days, f.recruiter_response_rate
        FROM candidates c
        LEFT JOIN candidate_features f ON c.id = f.candidate_id
        WHERE c.id IN ({placeholders})
    """
    cursor.execute(query, candidate_ids)
    
    features_map = {}
    for row in cursor.fetchall():
        features_map[row[0]] = {
            "skills": row[1].lower() if row[1] else "",
            "yoe": row[2] or 0.0,
            "ai_years": row[3] or 0.0,
            "job_hopping_index": row[4] or 0.0,
            "github_score": row[5] or 0.0,
            "education_tier": row[6] or 3,
            "notice_period": row[7] or 0,
            "response_rate": row[8] or 0.0
        }
        
    conn.close()
    
    dataset = []
    
    for cand_id, rrf_score in hybrid_results:
        feats = features_map.get(cand_id, {})
        
        # --- Honeypot Filtering ---
        # Discard mathematically impossible or extremely highly-suspicious profiles
        if feats.get("yoe", 0.0) > 50: continue
        if feats.get("notice_period", 0) > 180: continue
        # if feats.get("response_rate", 0.0) < 0.05: continue
        
        text = feats.get("skills", "")
        matched = sum(1 for skill in required_skills if skill in text)
        skill_score = matched / len(required_skills)
            
        dataset.append({
            "candidate_id": cand_id,
            "rrf_score": rrf_score,
            "skill_score": skill_score,
            "yoe": feats.get("yoe", 0.0),
            "ai_years": feats.get("ai_years", 0.0),
            "job_hopping_index": feats.get("job_hopping_index", 0.0),
            "github_score": feats.get("github_score", 0.0),
            "education_tier": feats.get("education_tier", 3),
            "notice_period": feats.get("notice_period", 0)
        })
        
    df = pd.DataFrame(dataset)
    
    # 3. Stage 4: Learning-to-Rank Inference
    print("Loading XGBoost Ranker model...")
    ranker = xgb.XGBRanker()
    ranker.load_model("models/xgb_ranker.json")
    
    features = [
        "rrf_score", "skill_score", "yoe", "ai_years", 
        "job_hopping_index", "github_score", "education_tier", "notice_period"
    ]
    
    X = df[features]
    df["score"] = ranker.predict(X)
    
    # Sort by ML score DESC, then by candidate_id ASC for tie-breaking
    df = df.sort_values(by=["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    
    # Keep only exact Top 100
    df = df.head(100)
    
    # Add Rank
    df["rank"] = range(1, 101)
    
    # Add Reasoning
    df["reasoning"] = df.apply(generate_reasoning, axis=1)
    
    # Output to Submission Format
    submission_df = df[["candidate_id", "rank", "score", "reasoning"]]
    
    # Format score to 4 decimal places for cleanliness
    submission_df.loc[:, "score"] = submission_df["score"].round(4)
    
    # Save to CSV
    output_filename = "team_submission.csv"
    submission_df.to_csv(output_filename, index=False, quoting=csv.QUOTE_MINIMAL)
    
    print(f"\nSuccessfully generated {output_filename} with EXACTLY 100 rows.")

if __name__ == "__main__":
    main()
