import pandas as pd
import sqlite3
import os
import sys

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

def generate_training_data():
    print("Loading Job Description...")
    doc = Document("data/jobs/job_description.docx")
    jd_text = "\n".join([para.text for para in doc.paragraphs])
    
    # We will fetch a large pool of candidates to create our training set (top 4000)
    top_n = 4000
    hybrid_results = get_hybrid_scores(jd_text, top_n=top_n)
    
    print("\nFetching features from database...")
    candidate_ids = [res[0] for res in hybrid_results]
    
    conn = sqlite3.connect("data/candidates.db")
    cursor = conn.cursor()
    
    placeholders = ",".join(["?"] * len(candidate_ids))
    query = f"""
        SELECT 
            c.id, c.skills, 
            f.yoe, f.ai_years, f.job_hopping_index, f.github_score, 
            f.notice_period_days, f.education_tier
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
            "notice_period": row[6] or 0,
            "education_tier": row[7] or 3
        }
        
    conn.close()
    
    dataset = []
    
    for cand_id, rrf_score in hybrid_results:
        feats = features_map.get(cand_id, {})
        text = feats.get("skills", "")
        
        matched = sum(1 for skill in required_skills if skill in text)
        skill_score = matched / len(required_skills)
        scaled_rrf = rrf_score * 30
        
        # Calculate Heuristic Score to use as a proxy for "Relevance"
        yoe = feats.get("yoe", 0.0)
        github = feats.get("github_score", 0.0)
        tier = feats.get("education_tier", 3)
        notice = feats.get("notice_period", 0)
        
        heuristic = (0.7 * scaled_rrf) + (0.3 * skill_score)
        heuristic += min(yoe, 10) * 0.01 
        heuristic += (github / 100.0) * 0.1
        if tier == 1: heuristic += 0.05
        if notice > 60: heuristic -= 0.10
            
        dataset.append({
            "candidate_id": cand_id,
            "rrf_score": rrf_score,
            "skill_score": skill_score,
            "yoe": yoe,
            "ai_years": feats.get("ai_years", 0.0),
            "job_hopping_index": feats.get("job_hopping_index", 0.0),
            "github_score": github,
            "education_tier": tier,
            "notice_period": notice,
            "heuristic_score": heuristic
        })
        
    df = pd.DataFrame(dataset)
    
    # Create relevance labels based on percentiles of the heuristic score
    # Top 5% -> 3 (Highly Relevant)
    # Next 15% -> 2 (Relevant)
    # Next 30% -> 1 (Somewhat Relevant)
    # Rest -> 0 (Not Relevant)
    q95 = df["heuristic_score"].quantile(0.95)
    q80 = df["heuristic_score"].quantile(0.80)
    q50 = df["heuristic_score"].quantile(0.50)
    
    def assign_label(score):
        if score >= q95: return 3
        if score >= q80: return 2
        if score >= q50: return 1
        return 0
        
    df["relevance"] = df["heuristic_score"].apply(assign_label)
    
    # Save training dataset
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/training_data.csv", index=False)
    print("Saved data/training_data.csv successfully!")
    print(df["relevance"].value_counts())

if __name__ == "__main__":
    generate_training_data()
