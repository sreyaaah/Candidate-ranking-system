import json
import sqlite3
import os
from tqdm import tqdm

DB_PATH = "data/candidates.db"
JSONL_PATH = "Dataset and references/candidates.jsonl"

def create_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE candidates (
            id TEXT PRIMARY KEY,
            full_text TEXT,
            skills TEXT
        )
    ''')
    
    conn.commit()
    return conn, cursor

def process_candidates():
    conn, cursor = create_db()
    
    print("Counting total candidates...")
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        total = sum(1 for _ in f)
        
    print(f"Found {total} candidates. Building database...")
    
    batch_size = 10000
    batch = []
    
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total):
            if not line.strip():
                continue
                
            data = json.loads(line)
            candidate_id = data.get("candidate_id", "")
            
            profile = data.get("profile", {})
            headline = profile.get("headline", "")
            summary = profile.get("summary", "")
            
            skills_list = data.get("skills", [])
            skills_text = ", ".join([s.get("name", "") if isinstance(s, dict) else str(s) for s in skills_list])
            
            # Priority 4: Weighted Candidate Sections (Repeat headline and skills to increase self-attention weights)
            weighted_headline = " ".join([f"Current Title: {headline}."] * 3)
            
            # Priority 5: Natural Language Skill Sentences
            skills_sentences = []
            for s in skills_list:
                if isinstance(s, dict):
                    name = s.get("name", "")
                    prof = s.get("proficiency", "intermediate")
                    dur = s.get("duration_months", 0)
                    skills_sentences.append(f"Candidate is proficient in {name} at an {prof} level for {dur} months.")
                else:
                    skills_sentences.append(f"Candidate has experience with {s}.")
            skills_rich_text = " ".join(skills_sentences)
            
            # Weight skills 2x
            weighted_skills = " ".join([skills_rich_text] * 2)
            
            # Career history with titles, companies, and descriptions (accomplishments)
            career_history = data.get("career_history", [])
            career_jobs = []
            for job in career_history:
                job_desc = job.get('description', '') or ""
                # Strip out newlines/extra spaces to keep clean
                job_desc_clean = " ".join(job_desc.split())
                job_str = f"Role: {job.get('title', '')} at {job.get('company', '')} for {job.get('duration_months', 0)} months. Accomplishments: {job_desc_clean}"
                career_jobs.append(job_str)
            career_text = " | ".join(career_jobs)
            
            # Education details
            edu_list = data.get("education", [])
            edu_items = []
            for edu in edu_list:
                edu_items.append(f"{edu.get('degree', '')} in {edu.get('field_of_study', '')} from {edu.get('institution', '')} (Grade: {edu.get('grade', '')})")
            edu_text = ", ".join(edu_items)
            
            # Certifications
            certs = data.get("certifications", [])
            cert_text = ", ".join([str(c) for c in certs])
            
            # Priority 3: Combined Rich Text with Structured Sections
            full_text = f"{weighted_headline} Profile Summary: {summary}. Core Competencies: {weighted_skills} Experience: {career_text}. Education: {edu_text}. Certifications: {cert_text}."
            
            # Keep skills field for faster skill matching later
            skills_lower = ", ".join([s.get("name", "") if isinstance(s, dict) else str(s) for s in skills_list]).lower()
            
            batch.append((candidate_id, full_text, skills_lower))
            
            if len(batch) >= batch_size:
                cursor.executemany("INSERT INTO candidates (id, full_text, skills) VALUES (?, ?, ?)", batch)
                batch = []
                
        if batch:
            cursor.executemany("INSERT INTO candidates (id, full_text, skills) VALUES (?, ?, ?)", batch)
            
    conn.commit()
    
    # Create an index for faster lookups
    cursor.execute('CREATE INDEX idx_candidate_id ON candidates(id)')
    conn.commit()
    
    conn.close()
    print(f"Database successfully saved to {DB_PATH}")

if __name__ == "__main__":
    process_candidates()
