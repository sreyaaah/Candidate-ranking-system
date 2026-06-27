import sqlite3
import json
import re
from tqdm import tqdm

DB_PATH = "data/candidates.db"
JSONL_PATH = "Dataset and references/candidates.jsonl"

def setup_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS candidate_features")
    cursor.execute('''
        CREATE TABLE candidate_features (
            candidate_id TEXT PRIMARY KEY,
            yoe REAL,
            ai_years REAL,
            job_hopping_index REAL,
            github_score REAL,
            notice_period_days INTEGER,
            recruiter_response_rate REAL,
            salary_min REAL,
            salary_max REAL,
            education_tier INTEGER,
            profile_completeness REAL
        )
    ''')
    conn.commit()
    return conn, cursor

def process_features():
    conn, cursor = setup_db()
    
    print("Counting total candidates for feature extraction...")
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        total = sum(1 for _ in f)
        
    print(f"Extracting features from {total} candidates...")
    
    batch = []
    batch_size = 5000
    
    ai_keywords = ['ai', 'ml', 'machine learning', 'data science', 'artificial intelligence', 'nlp', 'deep learning', 'llm', 'computer vision']
    
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total):
            if not line.strip():
                continue
                
            data = json.loads(line)
            candidate_id = data.get("candidate_id", "")
            if not candidate_id:
                continue
                
            # 1. Experience Features (YoE)
            profile = data.get("profile", {})
            yoe = profile.get("years_of_experience", 0.0)
            
            # 2. AI Years & Job Hopping
            career = data.get("career_history", [])
            ai_months = 0
            total_duration_months = 0
            job_count = len(career)
            
            for job in career:
                dur = job.get("duration_months") or 0
                total_duration_months += dur
                
                title = (job.get("title") or "").lower()
                desc = (job.get("description") or "").lower()
                
                if any(kw in title or kw in desc for kw in ai_keywords):
                    ai_months += dur
                    
            ai_years = round(ai_months / 12.0, 2)
            job_hopping_index = round(total_duration_months / job_count, 2) if job_count > 0 else 0.0
            
            # 3. Education Tier
            # Assuming tier_1 -> 1, tier_2 -> 2, tier_3 -> 3. (Lower is generally considered better in ranking, but we store the number)
            edu_list = data.get("education", [])
            best_tier = 3 # Default to tier 3
            for edu in edu_list:
                tier_str = edu.get("tier", "tier_3")
                if tier_str == "tier_1":
                    best_tier = min(best_tier, 1)
                elif tier_str == "tier_2":
                    best_tier = min(best_tier, 2)
            education_tier = best_tier
            
            # 4. Redrob Signals
            signals = data.get("redrob_signals", {})
            github_score = signals.get("github_activity_score", 0.0)
            notice_period_days = signals.get("notice_period_days", 0)
            recruiter_response_rate = signals.get("recruiter_response_rate", 0.0)
            profile_completeness = signals.get("profile_completeness_score", 0.0)
            
            salary = signals.get("expected_salary_range_inr_lpa", {})
            salary_min = salary.get("min", 0.0)
            salary_max = salary.get("max", 0.0)
            
            batch.append((
                candidate_id, yoe, ai_years, job_hopping_index, 
                github_score, notice_period_days, recruiter_response_rate,
                salary_min, salary_max, education_tier, profile_completeness
            ))
            
            if len(batch) >= batch_size:
                cursor.executemany("INSERT INTO candidate_features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
                batch = []
                
        if batch:
            cursor.executemany("INSERT INTO candidate_features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
            
    conn.commit()
    conn.close()
    
    print("Successfully extracted features into the 'candidate_features' table.")

if __name__ == "__main__":
    process_features()
