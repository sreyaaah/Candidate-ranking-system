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
from extract_skills import extract_required_skills, match_skills_with_ontology

def main():
    print("Loading Job Description...")
    doc = Document("data/jobs/job_description.docx")
    jd_text = "\n".join([para.text for para in doc.paragraphs])
    
    required_skills = extract_required_skills(jd_text)
    print(f"Dynamically extracted JD skills: {required_skills}")
    
    # 1. Stage 2: Hybrid Retrieval (Top 2000)
    top_n_hybrid = 2000
    hybrid_results = get_hybrid_scores(jd_text, top_n=top_n_hybrid)
    
    # 2. Stage 3: Fetch Features
    print("\nFetching features from database...")
    candidate_ids = [res[0] for res in hybrid_results]
    
    conn = sqlite3.connect("data/candidates.db")
    cursor = conn.cursor()
    
    placeholders = ",".join(["?"] * len(candidate_ids))
    query = f"""
        SELECT 
            c.id AS cand_db_id, c.skills, c.full_text, f.*
        FROM candidates c
        LEFT JOIN candidate_features f ON c.id = f.candidate_id
        WHERE c.id IN ({placeholders})
    """
    cursor.execute(query, candidate_ids)
    
    # Fetch all descriptions to get column headers
    col_names = [description[0] for description in cursor.description]
    
    features_map = {}
    for row in cursor.fetchall():
        row_dict = dict(zip(col_names, row))
        features_map[row_dict["cand_db_id"]] = row_dict
    conn.close()
    
    dataset = []
    for cand_id, rrf_score in hybrid_results:
        # --- Explicit Honeypot Filtering ---
        if cand_id == "CAND":
            continue
            
        feats = features_map.get(cand_id, {})
        if feats.get("honeypot_flag", 0) == 1:
            continue
        
        # Skill Ontology check
        candidate_skills = [s.strip() for s in str(feats.get("skills", "")).split(",") if s.strip()]
        matched_skills, missing_skills = match_skills_with_ontology(candidate_skills, required_skills)
        skill_score = len(matched_skills) / len(required_skills) if required_skills else 0.0
            
        feat_dict = {
            "candidate_id": cand_id,
            "full_text": feats.get("full_text", ""),
            "rrf_score": rrf_score,
            "skill_score": skill_score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills
        }
        
        # Copy over all numeric features from SQL mapped dict
        exclude_cols = {'id', 'skills', 'full_text', 'candidate_id'}
        for col, val in feats.items():
            if col not in exclude_cols:
                if pd.isna(val) or val is None:
                    val = 3.0 if col == 'education_tier' else 0.0
                feat_dict[col] = val
                
        dataset.append(feat_dict)
        
    df = pd.DataFrame(dataset)
    
    # 3. Stage 4: Learning-to-Rank Inference
    print("Loading XGBoost LTR Dual-Objective Ensemble...")
    ranker_pairwise = xgb.XGBRanker()
    ranker_pairwise.load_model("models/xgb_ranker_pairwise.json")
    
    ranker_ndcg = xgb.XGBRanker()
    ranker_ndcg.load_model("models/xgb_ranker_ndcg.json")
    
    exclude = {"cand_db_id", "candidate_id", "full_text", "matched_skills", "missing_skills", "ml_score", "ce_score", "norm_ml_score", "final_score", "score", "rank", "reasoning"}
    features = [col for col in df.columns if col not in exclude]
    
    X = df[features]
    
    # Predict with both LTR models
    scores_pw = ranker_pairwise.predict(X)
    scores_ndcg = ranker_ndcg.predict(X)
    
    # Min-max normalize predictions to blend them safely
    min_pw, max_pw = scores_pw.min(), scores_pw.max()
    norm_pw = (scores_pw - min_pw) / (max_pw - min_pw + 1e-9)
    
    min_ndcg, max_ndcg = scores_ndcg.min(), scores_ndcg.max()
    norm_ndcg = (scores_ndcg - min_ndcg) / (max_ndcg - min_ndcg + 1e-9)
    
    # Blend the ensemble
    df["ml_score"] = (norm_pw + norm_ndcg) / 2.0
    
    df = df.sort_values(by="ml_score", ascending=False).reset_index(drop=True)
    
    # Top 200 for Stage 5
    top_200 = df.head(200).copy()
    
    # 4. Stage 5: CrossEncoder Re-ranking
    import config
    print("\nLoading CrossEncoder for Stage 5 Top-N Re-ranking...")
    ce_model = CrossEncoder(config.CROSS_ENCODER_MODEL_NAME, max_length=512, local_files_only=True)
        
    # Give the model the title/context, plus the explicit requirements section
    jd_trunc = jd_text[:300] + "\n...[Requirements]...\n" + jd_text[3800:5200]
    
    print("Running CrossEncoder inference on Top 200 candidates...")
    ce_pairs = [(jd_trunc, str(txt)[:1000]) for txt in top_200["full_text"]]
    ce_scores = ce_model.predict(ce_pairs)
    
    # Normalize CE scores 0-1
    norm_ce_scores = 1 / (1 + np.exp(-ce_scores))
    top_200["ce_score"] = norm_ce_scores
    
    # Blend XGBoost (structural features) with CrossEncoder (semantic deep features)
    # Using Min-Max scaling for ML score to blend them nicely
    min_ml = top_200["ml_score"].min()
    max_ml = top_200["ml_score"].max()
    top_200["norm_ml_score"] = (top_200["ml_score"] - min_ml) / (max_ml - min_ml + 1e-9)
    
    top_200["final_score"] = (0.5 * top_200["norm_ml_score"]) + (0.5 * top_200["ce_score"])
    
    # Stage 6: MMR Diversity Re-ranking
    print("\nRunning Stage 6: MMR (Maximal Marginal Relevance) Diversity filtering...")
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(top_200["full_text"])
    
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    selected_indices = []
    unselected_indices = list(range(len(top_200)))
    
    # Select the very best candidate first
    best_idx = int(top_200["final_score"].idxmax())
    selected_indices.append(best_idx)
    unselected_indices.remove(best_idx)
    
    lambda_param = 0.85
    final_100_count = 100
    
    while len(selected_indices) < final_100_count and unselected_indices:
        max_mmr = -np.inf
        best_candidate_idx = -1
        
        for idx in unselected_indices:
            relevance = top_200.loc[idx, "final_score"]
            # Max similarity to already selected candidates
            sim_to_selected = max([sim_matrix[idx, s_idx] for s_idx in selected_indices])
            
            mmr_score = (lambda_param * relevance) - ((1 - lambda_param) * sim_to_selected)
            
            if mmr_score > max_mmr:
                max_mmr = mmr_score
                best_candidate_idx = idx
                
        selected_indices.append(best_candidate_idx)
        unselected_indices.remove(best_candidate_idx)
        
    final_top_100 = top_200.iloc[selected_indices].copy()
    
    # Create Structured Explainability 
    def generate_reasoning(row):
        conf = round(row['final_score'] * 100, 1)
        
        matched_str = ", ".join([f"✓ {s.capitalize()}" for s in row['matched_skills']]) if row['matched_skills'] else "None"
        missing_str = ", ".join([f"✗ {s.capitalize()}" for s in row['missing_skills']]) if row['missing_skills'] else "None"
        
        highlights = []
        if row['behavioral_score'] > 0: highlights.append("Leadership Experience")
        if row['company_fit_score'] > 0: highlights.append("Top Tier Tech Background")
        if row['education_tier'] == 1: highlights.append("Tier 1 Education")
        if row['career_trajectory_score'] > 0: highlights.append("Upward Career Trajectory")
        
        hl_str = ", ".join(highlights) if highlights else "Standard Profile"
        
        reasoning = f"Confidence: {conf}% | "
        reasoning += f"Matched: {matched_str} | "
        reasoning += f"Missing: {missing_str} | "
        reasoning += f"Highlights: {hl_str} | "
        
        risk = []
        if row['notice_period_days'] > 60:
            risk.append(f"High notice period ({int(row['notice_period_days'])} days)")
        if row['job_hopping_index'] < 12 and row['yoe'] > 3:
            risk.append("Frequent job hopper")
            
        if risk:
            reasoning += f"Risks: {', '.join(risk)}"
            
        return reasoning

    final_top_100["reasoning"] = final_top_100.apply(generate_reasoning, axis=1)
    
    # We must format to exactly: candidate_id, rank, score, reasoning
    # Round to 4 decimals BEFORE sorting to avoid floating point tie-breaker chaos
    final_top_100["score"] = final_top_100["final_score"].round(4)
    
    # Sort strictly by score DESC, candidate_id ASC for tie-breakers (per hackathon spec)
    final_top_100 = final_top_100.sort_values(by=["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    
    final_top_100["rank"] = final_top_100.index + 1
    
    submission_df = final_top_100[["candidate_id", "rank", "score", "reasoning"]]
    
    if len(submission_df) != 100:
        print(f"WARNING: Output has {len(submission_df)} rows, expected 100!")
        
    submission_df.to_csv("team_submission.csv", index=False)
    print("\nSuccessfully generated team_submission.csv with exactly 100 diverse rows.")

if __name__ == "__main__":
    main()
