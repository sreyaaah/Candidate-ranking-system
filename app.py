import sys

sys.path.append("src")
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import os

import sqlite3

st.title("AI Candidate Ranking System")

required_skills = [
    "python",
    "react",
    "node.js",
    "docker",
    "sql"
]

# Load embeddings
jd_embedding = np.load(
    "embeddings/jd_embedding.npy"
)

index = faiss.read_index(
    "embeddings/resume.index"
)

with open(
    "embeddings/resume_names.pkl",
    "rb"
) as f:
    resume_names = pickle.load(f)

k = min(2000, len(resume_names))
scores, indices = index.search(
    jd_embedding.reshape(1, -1),
    k
)

results = []

for i in range(len(indices[0])):

    idx = indices[0][i]

    resume_name = resume_names[idx]

    conn = sqlite3.connect("data/candidates.db")
    cursor = conn.cursor()
    cursor.execute("SELECT skills FROM candidates WHERE id=?", (resume_name,))
    row = cursor.fetchone()
    text = row[0].lower() if row else ""
    conn.close()

    matched = 0

    for skill in required_skills:
        if skill in text:
            matched += 1

    skill_score = matched / len(required_skills)

    semantic_score = scores[0][i]

    final_score = (
        0.7 * semantic_score
        + 0.3 * skill_score
    )

    results.append({
        "Candidate": resume_name,
        "Semantic Score": round(semantic_score, 4),
        "Skill Score": round(skill_score, 2),
        "Final Score": round(final_score, 4)
    })

df = pd.DataFrame(results)

df = df.sort_values(
    by="Final Score",
    ascending=False
)

st.subheader("Candidate Rankings")

st.dataframe(df)