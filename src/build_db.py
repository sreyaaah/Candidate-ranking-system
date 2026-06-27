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
            
            career_history = data.get("career_history", [])
            career_text = " ".join([
                f"{job.get('title', '')} at {job.get('company', '')}" 
                for job in career_history
            ])
            
            # Combine everything for embedding
            full_text = f"{headline}. {summary}. Skills: {skills_text}. Experience: {career_text}"
            
            # Keep skills field for faster skill matching later
            skills_lower = skills_text.lower()
            
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
