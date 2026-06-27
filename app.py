import sys
import os
import streamlit as st
import pandas as pd
import sqlite3
import xgboost as xgb
from docx import Document

sys.path.append("src")
from search_hybrid import get_hybrid_scores

st.set_page_config(layout="wide", page_title="INDIA.RUNS Candidate Ranking")
st.title("AI Candidate Ranking System - LTR")

required_skills = [
    "python",
    "react",
    "node.js",
    "docker",
    "sql"
]

@st.cache_data
def get_rankings():
    doc = Document("data/jobs/job_description.docx")
    jd_text = "\n".join([para.text for para in doc.paragraphs])
    
    top_n = 2000
    hybrid_results = get_hybrid_scores(jd_text, top_n=top_n)
    
    candidate_ids = [res[0] for res in hybrid_results]
    
    conn = sqlite3.connect("data/candidates.db")
    cursor = conn.cursor()
    placeholders = ",".join(["?"] * len(candidate_ids))
    
    query = f"""
        SELECT 
            c.id, c.skills, 
            f.yoe, f.ai_years, f.job_hopping_index, f.github_score, 
            f.education_tier, f.notice_period_days
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
            "notice_period": row[7] or 0
        }
    conn.close()
    
    dataset = []
    
    for cand_id, rrf_score in hybrid_results:
        feats = features_map.get(cand_id, {})
        text = feats.get("skills", "")
        
        matched = sum(1 for skill in required_skills if skill in text)
        skill_score = matched / len(required_skills)
            
        dataset.append({
            "Candidate ID": cand_id,
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
    
    ranker = xgb.XGBRanker()
    ranker.load_model("models/xgb_ranker.json")
    
    features = [
        "rrf_score", "skill_score", "yoe", "ai_years", 
        "job_hopping_index", "github_score", "education_tier", "notice_period"
    ]
    
    X = df[features]
    df["ML Score"] = ranker.predict(X)
    
    df = df.sort_values(by="ML Score", ascending=False).reset_index(drop=True)
    
    # Clean up column names for display
    df.rename(columns={
        "rrf_score": "RRF Score",
        "skill_score": "Skill Score",
        "yoe": "YoE",
        "ai_years": "AI Years",
        "job_hopping_index": "Job Hopping Index",
        "github_score": "GitHub Score",
        "education_tier": "Edu Tier",
        "notice_period": "Notice Period (Days)"
    }, inplace=True)
    
    def generate_ui_reasoning(row):
        yoe = float(row.get("YoE", 0.0))
        tier = int(row.get("Edu Tier", 3))
        notice = int(row.get("Notice Period (Days)", 0))
        github = float(row.get("GitHub Score", 0.0))
        skill = float(row.get("Skill Score", 0.0))
        ai_years = float(row.get("AI Years", 0.0))
        hop_index = float(row.get("Job Hopping Index", 0.0))
        
        tier_str = "Tier 1" if tier == 1 else "Tier 2" if tier == 2 else "Tier 3"
        
        reasons = []
        
        # Base Experience
        if ai_years > 0:
            reasons.append(f"Strong background with {yoe:.1f} YoE (including {ai_years:.1f} years focused on AI/ML).")
        else:
            reasons.append(f"Solid experience with {yoe:.1f} YoE.")
            
        # Education & Skills
        if skill > 0.6:
            reasons.append(f"Excellent keyword match for required skills (graduated from a {tier_str} institution).")
        else:
            reasons.append(f"Graduated from a {tier_str} institution with a decent skill baseline.")
            
        # Behavioral & Redrob Signals
        if github > 70:
            reasons.append(f"Demonstrates highly active technical engagement (GitHub: {github:.1f}).")
            
        if hop_index > 24:
            reasons.append("Shows great loyalty and career stability.")
        elif hop_index < 12 and hop_index > 0:
            reasons.append("Frequent job changes noted, but offset by strong technical fit.")
            
        # Notice Period
        if notice <= 30:
            reasons.append("Favorable notice period allows immediate onboarding.")
        elif notice > 60:
            reasons.append(f"Notice period of {notice} days is a minor logistical risk.")
            
        return " ".join(reasons)
        
    df["Reasoning"] = df.apply(generate_ui_reasoning, axis=1)
    
    return df

st.markdown("### Job Description Features")
st.write(f"**Required Skills:** {', '.join(required_skills)}")

with st.spinner('Running XGBoost Learning-to-Rank inference...'):
    df = get_rankings()

st.subheader(f"Top {len(df)} Ranked Candidates (ML Optimized)")
st.dataframe(df, use_container_width=True)