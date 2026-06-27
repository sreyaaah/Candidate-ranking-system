import numpy as np
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

def main():
    print("Loading Job Description...")
    doc = Document("data/jobs/job_description.docx")
    jd_text = "\n".join([para.text for para in doc.paragraphs])
    
    # Get Hybrid RRF scores
    top_n = 2000
    hybrid_results = get_hybrid_scores(jd_text, top_n=top_n)
    
    print("\nFetching skills from database...")
    candidate_ids = [res[0] for res in hybrid_results]
    
    # Bulk fetch to solve the N+1 query problem
    conn = sqlite3.connect("data/candidates.db")
    cursor = conn.cursor()
    
    placeholders = ",".join(["?"] * len(candidate_ids))
    cursor.execute(f"SELECT id, skills FROM candidates WHERE id IN ({placeholders})", candidate_ids)
    
    skills_map = {}
    for row in cursor.fetchall():
        skills_map[row[0]] = row[1].lower() if row[1] else ""
        
    conn.close()
    
    final_scores = []
    
    for cand_id, rrf_score in hybrid_results:
        text = skills_map.get(cand_id, "")
        
        matched = 0
        for skill in required_skills:
            if skill in text:
                matched += 1
                
        skill_score = matched / len(required_skills)
        
        # Combine RRF Score and Skill Score
        # Notice: RRF scores are typically small (e.g., 0.01-0.03) 
        # so we scale them or adjust the weights for final presentation
        scaled_rrf = rrf_score * 30 # roughly scale to a 0-1 range
        
        final_score = (0.7 * scaled_rrf) + (0.3 * skill_score)
        final_scores.append((cand_id, rrf_score, skill_score, final_score))
        
    # Re-sort by the final combined score
    final_scores.sort(key=lambda x: x[3], reverse=True)
    
    print("\nFinal Candidate Ranking (Top 20):\n")
    for i in range(min(20, len(final_scores))):
        cand_id, rrf, skill_s, final_s = final_scores[i]
        print(f"{i+1}. {cand_id}")
        print(f"RRF Score: {rrf:.4f} (Scaled: {rrf*30:.4f})")
        print(f"Skill Score: {skill_s:.2f}")
        print(f"Final Score: {final_s:.4f}\n")

if __name__ == "__main__":
    main()