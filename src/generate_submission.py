import numpy as np
import pandas as pd
import sqlite3
import os
import sys
import xgboost as xgb
import csv
from sentence_transformers import CrossEncoder

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
    skill = float(row.get("skill_score", 0.0))
    ai_years = float(row.get("ai_years", 0.0))
    hop_index = float(row.get("job_hopping_index", 0.0))
    
    tier_str = "Tier 1" if tier == 1 else "Tier 2" if tier == 2 else "Tier 3"
    
    reasons = []
    
    if ai_years > 0:
        reasons.append(f"Strong background with {yoe:.1f} YoE (including {ai_years:.1f} years focused on AI/ML).")
    else:
        reasons.append(f"Solid experience with {yoe:.1f} YoE.")
        
    if skill > 0.6:
        reasons.append(f"Excellent keyword match for required skills (graduated from a {tier_str} institution).")
    else:
        reasons.append(f"Graduated from a {tier_str} institution with a decent skill baseline.")
        
    if github > 70:
        reasons.append(f"Demonstrates highly active technical engagement (GitHub: {github:.1f}).")
        
    if hop_index > 24:
        reasons.append("Shows great loyalty and career stability.")
    elif hop_index < 12 and hop_index > 0:
        reasons.append("Frequent job changes noted, but offset by strong technical fit.")
        
    if notice <= 30:
        reasons.append("Favorable notice period allows immediate onboarding.")
    elif notice > 60:
        reasons.append(f"Notice period of {notice} days is a minor logistical risk.")
        
    return " ".join(reasons)

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
    # Note: Fetching full_text for the Cross-Encoder step
    query = f"""
        SELECT 
            c.id, c.skills, c.full_text,
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
            "full_text": row[2] or "",
            "yoe": row[3] or 0.0,
            "ai_years": row[4] or 0.0,
            "job_hopping_index": row[5] or 0.0,
            "github_score": row[6] or 0.0,
            "education_tier": row[7] or 3,
            "notice_period": row[8] or 0,
            "response_rate": row[9] or 0.0
        }
        
    conn.close()
    
    dataset = []
    
    for cand_id, rrf_score in hybrid_results:
        feats = features_map.get(cand_id, {})
        
        # --- Honeypot Filtering ---
        if feats.get("yoe", 0.0) > 50: continue
        if feats.get("notice_period", 0) > 180: continue
        
        text = feats.get("skills", "")
        matched = sum(1 for skill in required_skills if skill in text)
        skill_score = matched / len(required_skills)
            
        dataset.append({
            "candidate_id": cand_id,
            "full_text": feats.get("full_text", ""),
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
    df["ml_score"] = ranker.predict(X)
    
    # Sort by ML score
    df = df.sort_values(by="ml_score", ascending=False).reset_index(drop=True)
    
    # Keep Top 200 for Stage 5
    df_top_200 = df.head(200).copy()
    
    # 4. Stage 5: Cross-Encoder Re-Ranking
    print("\nLoading CrossEncoder for Stage 5 Top-N Re-ranking...")
    # Using a fast, highly accurate cross-encoder
    ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    
    # Prepare pairs: (JD, Candidate_Text)
    # We truncate JD and Candidate Text heavily to save compute time and fit context limits
    jd_trunc = jd_text[:1000]
    ce_pairs = [(jd_trunc, str(text)[:1000]) for text in df_top_200["full_text"]]
    
    print("Running CrossEncoder inference on Top 200 candidates...")
    ce_scores = ce_model.predict(ce_pairs)
    
    # Normalize CE scores between 0 and 1 using Sigmoid
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    norm_ce_scores = sigmoid(ce_scores)
    df_top_200["ce_score"] = norm_ce_scores
    
    # Blend ML Score and CE Score
    # We normalize ML score roughly as well to blend them
    min_ml = df_top_200["ml_score"].min()
    max_ml = df_top_200["ml_score"].max()
    norm_ml_scores = (df_top_200["ml_score"] - min_ml) / (max_ml - min_ml + 1e-9)
    
    df_top_200["final_score"] = (norm_ml_scores * 0.5) + (df_top_200["ce_score"] * 0.5)
    
    # Sort by Final Blended Score DESC, then candidate_id ASC for tie-breaking
    df_top_200 = df_top_200.sort_values(by=["final_score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    
    # Keep exact Top 100 for submission
    final_df = df_top_200.head(100).copy()
    
    # Add Rank
    final_df["rank"] = range(1, 101)
    
    # We must rename final_score to score for the validator
    final_df.rename(columns={"final_score": "score"}, inplace=True)
    
    # Add Reasoning
    final_df["reasoning"] = final_df.apply(generate_reasoning, axis=1)
    
    # Output to Submission Format
    submission_df = final_df[["candidate_id", "rank", "score", "reasoning"]]
    submission_df.loc[:, "score"] = submission_df["score"].round(4)
    
    output_filename = "team_submission.csv"
    submission_df.to_csv(output_filename, index=False, quoting=csv.QUOTE_MINIMAL)
    
    print(f"\nSuccessfully generated {output_filename} with EXACTLY 100 rows using Stage 5 Cross-Encoder.")

if __name__ == "__main__":
    main()
