import pandas as pd
import sqlite3
import re
import os

DB_PATH = "data/candidates.db"
CSV_PATH = "data/training_data.csv"

def extract_skills(full_text):
    if not isinstance(full_text, str):
        return ""
    # Look for "Skills: " or "Core Competencies: "
    match = re.search(r'Skills:\s*(.*)', full_text)
    if not match:
        match = re.search(r'Core Competencies:\s*(.*)', full_text)
    
    if match:
        skills_part = match.group(1)
        skills = []
        for s in skills_part.split(','):
            s = s.strip()
            if not s:
                continue
            # Remove parentheses e.g. (expert), (intermediate), (int...
            s_clean = re.sub(r'\s*\([^)]*\)?', '', s)
            s_clean = s_clean.strip()
            if s_clean:
                skills.append(s_clean)
        return ", ".join(skills).lower()
    return ""

def rebuild():
    print("Loading training data from CSV...")
    df = pd.read_csv(CSV_PATH)
    
    # Deduplicate by candidate_id to get unique candidate profiles
    df_unique = df.drop_duplicates(subset=["candidate_id"]).copy()
    print(f"Found {len(df_unique)} unique candidates.")
    
    # Remove existing db if any
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create candidates table
    print("Creating 'candidates' table...")
    cursor.execute('''
        CREATE TABLE candidates (
            id TEXT PRIMARY KEY,
            full_text TEXT,
            skills TEXT
        )
    ''')
    
    # Populate candidates table
    candidates_data = []
    for _, row in df_unique.iterrows():
        cand_id = row["candidate_id"]
        full_text = row["full_text"]
        skills = extract_skills(full_text)
        candidates_data.append((cand_id, full_text, skills))
        
    cursor.executemany("INSERT INTO candidates (id, full_text, skills) VALUES (?, ?, ?)", candidates_data)
    cursor.execute('CREATE INDEX idx_candidate_id ON candidates(id)')
    conn.commit()
    print("Successfully populated 'candidates' table.")
    
    # Create candidate_features table
    print("Creating 'candidate_features' table...")
    features_columns = [
        "candidate_id", "yoe", "ai_years", "ai_continuity", "job_count", "job_hopping_index",
        "promotion_velocity", "retrieval_exp", "vectordb_exp", "embedding_exp",
        "product_experience", "leadership_score", "github_score", "github_active",
        "education_tier", "notice_period_days", "recruiter_response_rate", "profile_completeness",
        "salary_min", "salary_max", "company_fit_score", "career_trajectory_score",
        "behavioral_score", "location_match", "honeypot_flag", "certifications_count",
        "languages_count", "skills_count", "avg_skill_duration", "highest_endorsement",
        "academic_performance", "recent_ai_experience", "open_source_contributor",
        "cloud_experience", "big_data_experience", "mlops_experience", "llm_experience",
        "deep_learning_exp", "seniority_level", "tenure_current_job", "career_gap",
        "profile_views_received_30d", "avg_response_time_hours", "connection_count",
        "endorsements_received", "interview_completion_rate", "offer_acceptance_rate"
    ]
    
    # Create table SQL
    sql_cols = []
    for col in features_columns:
        if col == "candidate_id":
            sql_cols.append(f"{col} TEXT PRIMARY KEY")
        elif col in ["job_count", "retrieval_exp", "vectordb_exp", "embedding_exp", "github_active", 
                     "education_tier", "notice_period_days", "location_match", "honeypot_flag", 
                     "certifications_count", "languages_count", "skills_count", "highest_endorsement", 
                     "recent_ai_experience", "open_source_contributor", "cloud_experience", 
                     "big_data_experience", "mlops_experience", "llm_experience", "deep_learning_exp", 
                     "seniority_level", "profile_views_received_30d", "connection_count", 
                     "endorsements_received"]:
            sql_cols.append(f"{col} INTEGER")
        else:
            sql_cols.append(f"{col} REAL")
            
    create_sql = f"CREATE TABLE candidate_features ({', '.join(sql_cols)})"
    cursor.execute(create_sql)
    
    # Populate candidate_features table
    print("Populating 'candidate_features' table...")
    features_data = []
    for _, row in df_unique.iterrows():
        row_values = []
        for col in features_columns:
            val = row[col]
            if pd.isna(val) or val is None:
                val = 3.0 if col == 'education_tier' else 0.0
            row_values.append(val)
        features_data.append(tuple(row_values))
        
    placeholders = ",".join(["?"] * len(features_columns))
    cursor.executemany(f"INSERT INTO candidate_features VALUES ({placeholders})", features_data)
    conn.commit()
    conn.close()
    
    print("Successfully populated 'candidate_features' table.")
    print("Database rebuilding completed successfully!")

if __name__ == "__main__":
    rebuild()
