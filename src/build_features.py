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
            ai_continuity REAL,
            job_count INTEGER,
            job_hopping_index REAL,
            promotion_velocity REAL,
            retrieval_exp INTEGER,
            vectordb_exp INTEGER,
            embedding_exp INTEGER,
            product_experience REAL,
            leadership_score REAL,
            github_score REAL,
            github_active INTEGER,
            education_tier INTEGER,
            notice_period_days INTEGER,
            recruiter_response_rate REAL,
            profile_completeness REAL,
            salary_min REAL,
            salary_max REAL,
            company_fit_score REAL,
            career_trajectory_score REAL,
            behavioral_score REAL,
            location_match INTEGER,
            honeypot_flag INTEGER,
            certifications_count INTEGER,
            languages_count INTEGER,
            skills_count INTEGER,
            avg_skill_duration REAL,
            highest_endorsement INTEGER,
            academic_performance REAL,
            recent_ai_experience INTEGER,
            open_source_contributor INTEGER,
            cloud_experience INTEGER,
            big_data_experience INTEGER,
            mlops_experience INTEGER,
            llm_experience INTEGER,
            deep_learning_exp INTEGER,
            seniority_level INTEGER,
            tenure_current_job REAL,
            career_gap REAL,
            profile_views_received_30d INTEGER,
            avg_response_time_hours REAL,
            connection_count INTEGER,
            endorsements_received INTEGER,
            interview_completion_rate REAL,
            offer_acceptance_rate REAL
        )
    ''')
    conn.commit()
    return conn, cursor

def process_features():
    conn, cursor = setup_db()
    
    print("Counting total candidates for feature extraction...")
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        total = sum(1 for _ in f)
        
    print(f"Extracting 46 features from {total} candidates...")
    
    batch = []
    batch_size = 5000
    
    ai_keywords = ['ai', 'ml', 'machine learning', 'data science', 'artificial intelligence', 'nlp', 'deep learning', 'llm', 'computer vision']
    faang_keywords = ['google', 'amazon', 'facebook', 'meta', 'apple', 'netflix', 'microsoft', 'adobe', 'uber']
    behavioral_keywords = ['led', 'managed', 'mentored', 'coordinated', 'spearheaded', 'directed']
    
    # Specific targeted domains
    retrieval_keywords = ['retrieval', 'search', 'bm25', 'information retrieval', 'hybrid search', 'elastic', 'solr', 'lucene']
    vectordb_keywords = ['milvus', 'pinecone', 'chroma', 'weaviate', 'qdrant', 'faiss', 'vespa']
    embedding_keywords = ['embedding', 'sentence-transformer', 'encoder', 'bert', 'transformers']
    llm_keywords = ['llm', 'langchain', 'llama', 'gpt', 'rag', 'prompt engineering']
    mlops_keywords = ['mlops', 'docker', 'kubernetes', 'airflow', 'mlflow', 'dvc']
    dl_keywords = ['pytorch', 'tensorflow', 'keras', 'deep learning', 'neural network']
    bigdata_keywords = ['spark', 'hadoop', 'kafka', 'flink', 'hive']
    cloud_keywords = ['aws', 'azure', 'gcp', 'cloud', 's3', 'ec2']
    consulting_keywords = ['consulting', 'services', 'systems', 'infosys', 'wipro', 'tcs', 'cognizant', 'accenture', 'capgemini']
    
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total):
            if not line.strip():
                continue
                
            data = json.loads(line)
            candidate_id = data.get("candidate_id", "")
            if not candidate_id:
                continue
                
            # 1. Profile / Experience Features
            profile = data.get("profile", {})
            yoe = profile.get("years_of_experience", 0.0)
            
            loc = (profile.get("location") or "").lower()
            location_match = 1 if "india" in loc or "remote" in loc else 0
            
            # 2. Career History
            career = data.get("career_history", [])
            job_count = len(career)
            
            ai_months = 0
            total_duration_months = 0
            
            retrieval_exp = 0
            vectordb_exp = 0
            embedding_exp = 0
            llm_experience = 0
            mlops_experience = 0
            deep_learning_exp = 0
            big_data_experience = 0
            cloud_experience = 0
            
            company_fit_score = 0.0
            behavioral_score = 0.0
            leadership_score = 0.0
            product_jobs = 0
            
            has_junior = False
            has_senior = False
            
            promotions = 0
            recent_ai_experience = 0
            open_source_contributor = 0
            tenure_current_job = 0.0
            career_gap = 0.0
            seniority_level = 0
            
            # Temporary store for gaps
            job_dates = []
            
            for idx, job in enumerate(career):
                dur = job.get("duration_months") or 0
                total_duration_months += dur
                
                title = (job.get("title") or "").lower()
                desc = (job.get("description") or "").lower()
                company = (job.get("company") or "").lower()
                
                # Check recent job
                if idx == 0:
                    tenure_current_job = float(dur)
                    if any(kw in title or kw in desc for kw in ai_keywords):
                        recent_ai_experience = 1
                    if any(kw in title for kw in ['lead', 'manager', 'principal', 'chief', 'director', 'vp', 'architect']):
                        seniority_level = 3
                    elif any(kw in title for kw in ['senior', 'sr.']):
                        seniority_level = 2
                    elif any(kw in title for kw in ['junior', 'jr.', 'intern']):
                        seniority_level = 0
                    else:
                        seniority_level = 1
                
                # Tech keywords
                if any(kw in title or kw in desc for kw in ai_keywords):
                    ai_months += dur
                if any(kw in title or kw in desc for kw in retrieval_keywords):
                    retrieval_exp = 1
                if any(kw in title or kw in desc for kw in vectordb_keywords):
                    vectordb_exp = 1
                if any(kw in title or kw in desc for kw in embedding_keywords):
                    embedding_exp = 1
                if any(kw in title or kw in desc for kw in llm_keywords):
                    llm_experience = 1
                if any(kw in title or kw in desc for kw in mlops_keywords):
                    mlops_experience = 1
                if any(kw in title or kw in desc for kw in dl_keywords):
                    deep_learning_exp = 1
                if any(kw in title or kw in desc for kw in bigdata_keywords):
                    big_data_experience = 1
                if any(kw in title or kw in desc for kw in cloud_keywords):
                    cloud_experience = 1
                    
                # Open source
                if "open-source" in desc or "contributed to" in desc or "github repository" in desc:
                    open_source_contributor = 1
                    
                # Company types
                if any(kw in company for kw in faang_keywords):
                    company_fit_score = 1.0
                if not any(kw in company for kw in consulting_keywords):
                    product_jobs += 1
                    
                # Behavioral & Leadership
                if any(kw in desc for kw in behavioral_keywords):
                    behavioral_score += 1.0
                if any(kw in title for kw in ['lead', 'manager', 'principal', 'chief', 'director', 'vp', 'architect']):
                    leadership_score += 1.0
                    
                if "junior" in title or "intern" in title:
                    has_junior = True
                if "senior" in title or "lead" in title or "manager" in title:
                    has_senior = True
                    
            # Promotion estimation: check adjacent jobs for same company
            for i in range(len(career) - 1):
                if career[i].get("company", "").lower() == career[i+1].get("company", "").lower():
                    promotions += 1
            
            ai_years = round(ai_months / 12.0, 2)
            ai_continuity = round(ai_years / yoe, 2) if yoe > 0 else 0.0
            job_hopping_index = round(total_duration_months / job_count, 2) if job_count > 0 else 0.0
            
            career_trajectory_score = 1.0 if (has_junior and has_senior) else 0.5 if has_senior else 0.0
            behavioral_score = min(behavioral_score / 3.0, 1.0)
            product_experience = round(product_jobs / job_count, 2) if job_count > 0 else 0.0
            promotion_velocity = round(promotions / job_count, 2) if job_count > 0 else 0.0
            
            # 3. Education
            edu_list = data.get("education", [])
            best_tier = 3 
            academic_performance = 0.0
            for edu in edu_list:
                tier_str = edu.get("tier", "tier_3")
                if tier_str == "tier_1":
                    best_tier = min(best_tier, 1)
                elif tier_str == "tier_2":
                    best_tier = min(best_tier, 2)
                
                grade = str(edu.get("grade", "")).lower()
                if "cgpa" in grade or "gpa" in grade:
                    # extract numbers
                    match = re.search(r"(\d+\.\d+)", grade)
                    if match:
                        gpa = float(match.group(1))
                        # Normalize to 0-1 scale assuming 10 or 4 base
                        if gpa > 4.0:
                            academic_performance = max(academic_performance, gpa / 10.0)
                        else:
                            academic_performance = max(academic_performance, gpa / 4.0)
                elif "%" in grade or "percent" in grade:
                    match = re.search(r"(\d+)", grade)
                    if match:
                        academic_performance = max(academic_performance, float(match.group(1)) / 100.0)
            
            education_tier = best_tier
            
            # 4. Skills & Certifications
            skills_list = data.get("skills", [])
            skills_count = len(skills_list)
            
            total_skill_duration = sum(s.get("duration_months") or 0 for s in skills_list if isinstance(s, dict))
            avg_skill_duration = round(total_skill_duration / skills_count, 2) if skills_count > 0 else 0.0
            highest_endorsement = max([s.get("endorsements") or 0 for s in skills_list if isinstance(s, dict)] + [0])
            
            certs = data.get("certifications", [])
            certifications_count = len(certs)
            
            languages = data.get("languages", [])
            languages_count = len(languages)
            
            # 5. Redrob Signals
            signals = data.get("redrob_signals", {})
            github_score = signals.get("github_activity_score", 0.0)
            github_active = 1 if github_score > 0 else 0
            
            notice_period_days = signals.get("notice_period_days", 0)
            recruiter_response_rate = signals.get("recruiter_response_rate", 0.0)
            profile_completeness = signals.get("profile_completeness_score", 0.0)
            
            salary = signals.get("expected_salary_range_inr_lpa", {})
            salary_min = salary.get("min", 0.0)
            salary_max = salary.get("max", 0.0)
            
            # Additional signals
            profile_views_received_30d = signals.get("profile_views_received_30d", 0)
            avg_response_time_hours = signals.get("avg_response_time_hours", 0.0)
            connection_count = signals.get("connection_count", 0)
            endorsements_received = signals.get("endorsements_received", 0)
            interview_completion_rate = signals.get("interview_completion_rate", 0.0)
            offer_acceptance_rate = signals.get("offer_acceptance_rate", 0.0)
            
            honeypot_flag = 1 if (yoe > 50 or notice_period_days > 180 or recruiter_response_rate < 0.01) else 0
            
            batch.append((
                candidate_id, yoe, ai_years, ai_continuity, job_count, job_hopping_index,
                promotion_velocity, retrieval_exp, vectordb_exp, embedding_exp,
                product_experience, leadership_score, github_score, github_active,
                education_tier, notice_period_days, recruiter_response_rate, profile_completeness,
                salary_min, salary_max, company_fit_score, career_trajectory_score,
                behavioral_score, location_match, honeypot_flag, certifications_count,
                languages_count, skills_count, avg_skill_duration, highest_endorsement,
                academic_performance, recent_ai_experience, open_source_contributor,
                cloud_experience, big_data_experience, mlops_experience, llm_experience,
                deep_learning_exp, seniority_level, tenure_current_job, career_gap,
                profile_views_received_30d, avg_response_time_hours, connection_count,
                endorsements_received, interview_completion_rate, offer_acceptance_rate
            ))
            
            if len(batch) >= batch_size:
                cursor.executemany("INSERT INTO candidate_features VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                batch = []
                
        if batch:
            cursor.executemany("INSERT INTO candidate_features VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
            
    conn.commit()
    conn.close()
    
    print("Successfully extracted 46 features into the 'candidate_features' table.")

if __name__ == "__main__":
    process_features()
