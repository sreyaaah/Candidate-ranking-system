import numpy as np
import pandas as pd
import sqlite3
import os
import sys
import xgboost as xgb
import csv
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
        
    # Stage 6 explicit mention
    if row.get("diversity_promoted", False):
        reasons.append("Candidate was actively promoted by the MMR algorithm to ensure structural team diversity.")
        
    return " ".join(reasons)

def main():
    print("Loading Job Description...")
    doc = Document("data/jobs/job_description.docx")
    jd_text = "\n".join([para.text for para in doc.paragraphs])
    
    # 1. Stage 1 & 2: Hybrid Retrieval
    top_n = 2000
    hybrid_results = get_hybrid_scores(jd_text, top_n=top_n)
    
    print("\nFetching features from database...")
    candidate_ids = [res[0] for res in hybrid_results]
    
    conn = sqlite3.connect("data/candidates.db")
    cursor = conn.cursor()
    
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
            "response_rate": row[9] or 0.0,
            "company_fit_score": row[10] or 0.0,
            "career_trajectory_score": row[11] or 0.0,
            "behavioral_score": row[12] or 0.0,
            "location_match": row[13] or 0,
            "honeypot_flag": row[14] or 0
        }
    conn.close()
    
    dataset = []
    for cand_id, rrf_score in hybrid_results:
        feats = features_map.get(cand_id, {})
        # --- Honeypot Filtering ---
        if feats.get("honeypot_flag", 0) == 1: continue
        
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
            "notice_period": feats.get("notice_period", 0),
            "company_fit_score": feats.get("company_fit_score", 0.0),
            "career_trajectory_score": feats.get("career_trajectory_score", 0.0),
            "behavioral_score": feats.get("behavioral_score", 0.0),
            "location_match": feats.get("location_match", 0),
            "honeypot_flag": feats.get("honeypot_flag", 0)
        })
        
    df = pd.DataFrame(dataset)
    
    # 3. Stage 4: Learning-to-Rank Inference
    print("Loading XGBoost Ranker model...")
    ranker = xgb.XGBRanker()
    ranker.load_model("models/xgb_ranker.json")
    features = [
        "rrf_score", "skill_score", "yoe", "ai_years", 
        "job_hopping_index", "github_score", "education_tier", "notice_period",
        "company_fit_score", "career_trajectory_score", "behavioral_score",
        "location_match", "honeypot_flag"
    ]
    X = df[features]
    df["ml_score"] = ranker.predict(X)
    df = df.sort_values(by="ml_score", ascending=False).reset_index(drop=True)
    df_top_200 = df.head(200).copy()
    
    # 4. Stage 5: Cross-Encoder Re-Ranking
    print("\nLoading CrossEncoder for Stage 5 Top-N Re-ranking...")
    ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    jd_trunc = jd_text[:1000]
    ce_pairs = [(jd_trunc, str(text)[:1000]) for text in df_top_200["full_text"]]
    
    print("Running CrossEncoder inference on Top 200 candidates...")
    ce_scores = ce_model.predict(ce_pairs)
    norm_ce_scores = 1 / (1 + np.exp(-ce_scores)) # Sigmoid
    
    min_ml = df_top_200["ml_score"].min()
    max_ml = df_top_200["ml_score"].max()
    norm_ml_scores = (df_top_200["ml_score"] - min_ml) / (max_ml - min_ml + 1e-9)
    df_top_200["final_score"] = (norm_ml_scores * 0.5) + (norm_ce_scores * 0.5)
    
    # 5. Stage 6: Maximal Marginal Relevance (Diversity)
    print("\nRunning Stage 6: MMR (Maximal Marginal Relevance) Diversity filtering...")
    texts = df_top_200["full_text"].tolist()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    # Extract needed arrays for fast computation
    candidates = df_top_200.to_dict('records')
    scores = np.array([c["final_score"] for c in candidates])
    
    selected_indices = []
    unselected_indices = list(range(len(candidates)))
    
    LAMBDA = 0.85 # 85% relevance, 15% diversity penalty
    
    while len(selected_indices) < 100 and unselected_indices:
        if not selected_indices:
            # First item is purely the most relevant
            best_idx = unselected_indices[np.argmax(scores[unselected_indices])]
        else:
            # MMR formula
            unsel_scores = scores[unselected_indices]
            # Max similarity to ANY already selected candidate
            sim_to_selected = sim_matrix[unselected_indices][:, selected_indices]
            max_sims = np.max(sim_to_selected, axis=1)
            
            mmr_scores = (LAMBDA * unsel_scores) - ((1 - LAMBDA) * max_sims)
            best_idx_in_unselected = np.argmax(mmr_scores)
            best_idx = unselected_indices[best_idx_in_unselected]
            
            # Check if this candidate was promoted purely due to MMR
            pure_relevance_idx = unselected_indices[np.argmax(unsel_scores)]
            if best_idx != pure_relevance_idx:
                candidates[best_idx]["diversity_promoted"] = True
                
        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)
        
    final_candidates = [candidates[i] for i in selected_indices]
    final_df = pd.DataFrame(final_candidates)
    
    # We must rename final_score to score for the validator
    final_df.rename(columns={"final_score": "score"}, inplace=True)
    
    # Round score BEFORE sorting so tie-breakers are accurate for the CSV
    final_df["score"] = final_df["score"].round(4)
    
    # Format tie breaking securely
    final_df = final_df.sort_values(by=["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    final_df["rank"] = range(1, 101)
    
    # Add Reasoning
    final_df["reasoning"] = final_df.apply(generate_reasoning, axis=1)
    
    # Output to Submission Format
    submission_df = final_df[["candidate_id", "rank", "score", "reasoning"]]
    
    output_filename = "team_submission.csv"
    submission_df.to_csv(output_filename, index=False, quoting=csv.QUOTE_MINIMAL)
    
    print(f"\nSuccessfully generated {output_filename} with exactly 100 diverse rows.")

if __name__ == "__main__":
    main()
